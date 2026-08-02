from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import click
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 将项目根目录加入 sys.path，使 skills/ 可导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_react import ReActAgent
from console.commands import (
    COMMAND_SPECS,
    MODEL_CHOICES,
    banner,
    handle_command,
    help_text,
    model_lines,
)
from console.readline import create_readline
from config import AppConfig, load_config
from core import get_logger
from llm import LLMClient, LocalEmbedder
from memory import MemoryBundle, MemoryPolicy, MemoryStore
from reporter import ProgressReporter
from retrieval import Retriever
from skill_manager import SkillManager
from skills.web_search.client import WebSearchClient
from search_config import ensure_search_config

log = get_logger("auraderma.cli")

# ======================================================================
# 运行时构建
# ======================================================================


@dataclass(slots=True)
class Runtime:
    """一次构建产出的所有运行时组件。"""
    react_agent: ReActAgent
    cfg: AppConfig
    model: str
    store: MemoryStore
    skills: SkillManager
    embedder: LocalEmbedder
    retriever: Retriever
    web: WebSearchClient
    policy: MemoryPolicy


def _load_env() -> None:
    load_dotenv()
    load_dotenv(".env.local", override=True)


def _build_runtime(
    model_override: str | None = None,
) -> Runtime:
    _load_env()
    ensure_search_config()
    warnings.filterwarnings("ignore", message=".*incompatible with server version.*")
    cfg = load_config()

    # 优先连接远程 Qdrant，失败则自动回退到本地持久化模式
    if cfg.qdrant_api_key:
        qdrant = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    else:
        try:
            qdrant = QdrantClient(url=cfg.qdrant_url, timeout=2)
            qdrant.get_collections()  # 探活
        except Exception:
            log.warning("Qdrant 远程服务不可用（%s），回退到本地模式", cfg.qdrant_url)
            qdrant = QdrantClient(path=str(cfg.data_dir / "qdrant"))

    embedder = LocalEmbedder()

    retriever = Retriever(
        qdrant,
        cfg.qdrant_collection_products,
        cfg.qdrant_collection_memory,
        cfg.qdrant_collection_docs,
    )
    retriever.ensure_collections(vector_size=embedder.dim)
    llm = LLMClient(
        api_key=cfg.model_api_key,
        base_url=cfg.model_api_base,
        model=model_override or cfg.default_model,
    )

    web = WebSearchClient(enabled=cfg.web_search_enabled)
    reporter = ProgressReporter()

    react_agent = ReActAgent(
        llm=llm,
        retriever=retriever,
        embedder=embedder,
        web=web,
        reporter=reporter,
    )
    store = MemoryStore(cfg.data_dir / "memory")
    policy = MemoryPolicy()
    skills = SkillManager(cfg.skills_dir, web)
    return Runtime(
        react_agent=react_agent,
        cfg=cfg,
        model=llm.model,
        store=store,
        skills=skills,
        embedder=embedder,
        retriever=retriever,
        web=web,
        policy=policy,
    )


# ======================================================================
# AppSession
# ======================================================================


class AppSession:
    def __init__(self, model_override: str | None = None):
        rt = _build_runtime(model_override)
        self.react_agent = rt.react_agent
        self.cfg = rt.cfg
        self.current_model = rt.model
        self.memory_store = rt.store
        self.skills = rt.skills
        self.embedder = rt.embedder
        self.retriever = rt.retriever
        self.web = rt.web
        self.policy = rt.policy

    def refresh_model(self, model_name: str) -> None:
        rt = _build_runtime(model_name)
        self.react_agent = rt.react_agent
        self.cfg = rt.cfg
        self.current_model = rt.model
        self.memory_store = rt.store
        self.skills = rt.skills
        self.embedder = rt.embedder
        self.retriever = rt.retriever
        self.web = rt.web
        self.policy = rt.policy


# ======================================================================
# CLI 入口
# ======================================================================


@click.group()
def main() -> None:
    """AuraDerma CLI (ReAct Agent)."""


@main.command()
@click.option("--model", type=str, default=None, help="Override model name for this session")
@click.option("--user-id", default="default-user")
def chat(model: str | None, user_id: str) -> None:
    """启动交互式聊天会话（基于 ReAct Agent 的思考-行动-观察循环）。"""
    session = AppSession(model)
    memory = session.memory_store.load(user_id)
    turn_log: list[str] = []
    click.echo(banner(session.current_model))
    sys.stdout.flush()

    history: list[str] = []
    command_names = [spec.name for spec in COMMAND_SPECS]
    command_descriptions = [spec.description for spec in COMMAND_SPECS]

    readline_fn = create_readline(
        session.current_model,
        command_names,
        command_descriptions,
        history,
    )

    while True:
        try:
            raw = readline_fn()
        except NotImplementedError:
            log.info("当前平台不支持高级 readline，降级到简单输入模式")
            _fallback_chat(session, user_id, memory, turn_log)
            return

        if raw is None:
            click.echo("bye.")
            break

        cmd = raw.strip()
        if cmd.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if not cmd:
            continue
        if cmd.startswith("/"):
            handled = handle_command(cmd, session, user_id, memory)
            if handled == "quit":
                break
            if handled == "switched":
                click.echo(banner(session.current_model))
            continue

        # ── ReAct 思考→行动→观察循环 ──
        answer = _do_chat(cmd, user_id, session, memory, turn_log)
        click.echo("\n[assistant]\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")


def _do_chat(
    cmd: str,
    user_id: str,
    session: AppSession,
    memory: MemoryBundle,
    turn_log: list[str],
) -> str:
    """执行一次 ReAct 对话（思考→行动→观察循环）。"""
    result = session.react_agent.run(
        question=cmd,
        user_id=user_id,
        memory_store=session.memory_store,
    )
    # 打印工具调用步骤
    if result.steps:
        click.echo(f"\n[ReAct: {result.total_tool_calls} 步]")
        for s in result.steps:
            click.echo(f"  #{s['step']} {s['tool']}({s['args']}) → {s['result_preview'][:60]}...")

    # 重新加载记忆（ReActAgent 已在内部写入新记忆）
    fresh = session.memory_store.load(user_id)
    memory.profile = fresh.profile
    memory.short_term = fresh.short_term
    memory.long_term = fresh.long_term
    memory.case_notes = fresh.case_notes
    memory.short_term.append(
        session.policy.classify(f"本轮对话：{cmd}", user_id=user_id)
    )
    return result.answer


def _fallback_chat(
    session: AppSession,
    user_id: str,
    memory: MemoryBundle,
    turn_log: list[str],
) -> None:
    """降级到简单的 input 模式（所有平台通用）。"""
    click.echo("fallback to simple input mode.")

    while True:
        try:
            raw = click.prompt(f"AuraDerma[{session.current_model}]", type=str)
        except (EOFError, KeyboardInterrupt):
            break
        cmd = raw.strip()
        if cmd.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if cmd.startswith("/"):
            handled = handle_command(cmd, session, user_id, memory)
            if handled == "quit":
                break
            continue

        answer = _do_chat(cmd, user_id, session, memory, turn_log)
        click.echo("\nassistant:\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")


@main.command()
def init() -> None:
    """显示初始配置信息。"""
    _load_env()
    cfg = load_config()
    click.echo(
        json.dumps(
            {
                "model_base": cfg.model_api_base,
                "default_model": cfg.default_model,
                "qdrant_url": cfg.qdrant_url,
                "skills_dir": str(cfg.skills_dir),
                "collections": [
                    cfg.qdrant_collection_products,
                    cfg.qdrant_collection_memory,
                    cfg.qdrant_collection_docs,
                ],
                "models": MODEL_CHOICES,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@main.command()
@click.argument("path", type=click.Path(path_type=Path, exists=True))
def ingest(path: Path) -> None:
    """导入文档（PDF/DOCX/TXT）。"""
    from ingest import ingest_document

    result = ingest_document(path)
    click.echo(
        json.dumps(
            {
                "doc_id": result.doc_id,
                "path": str(result.path),
                "type": result.doc_type.value,
                "chunks": len(result.chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@main.command()
@click.argument("path", type=click.Path(path_type=Path, exists=True))
def ingest_products(path: Path) -> None:
    """解析护肤品原始文本文件，结构化后 embedding 存入 Qdrant。"""
    _load_env()
    cfg = load_config()
    qdrant = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    retriever = Retriever(
        qdrant,
        cfg.qdrant_collection_products,
        cfg.qdrant_collection_memory,
        cfg.qdrant_collection_docs,
    )
    retriever.ensure_collections()

    from ingest_products import ingest_products_to_qdrant

    embedder = LocalEmbedder()
    records = ingest_products_to_qdrant(path, embedder, retriever)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "file": str(path),
                "products_ingested": len(records),
                "products": [
                    {
                        "id": r.product_id,
                        "name": r.name,
                        "brand": r.brand,
                        "concerns": r.concerns,
                    }
                    for r in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
