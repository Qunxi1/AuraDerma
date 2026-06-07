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

from agent import AgentContext, SkincareAgent
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
from llm import LLMClient
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


def _load_env() -> None:
    load_dotenv()
    load_dotenv(".env.local", override=True)


def _build_runtime(
    model_override: str | None = None,
) -> tuple[SkincareAgent, AppConfig, str, MemoryStore, SkillManager]:
    _load_env()
    ensure_search_config()
    warnings.filterwarnings("ignore", message=".*incompatible with server version.*")
    cfg = load_config()
    qdrant = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    from llm import LocalEmbedder

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
    policy = MemoryPolicy()
    reporter = ProgressReporter()
    agent = SkincareAgent(
        llm=llm,
        retriever=retriever,
        web=web,
        policy=policy,
        _embedder=embedder,
        reporter=reporter,
    )
    store = MemoryStore(cfg.data_dir / "memory")
    skills = SkillManager(cfg.skills_dir, web)
    return agent, cfg, llm.model, store, skills


# ======================================================================
# AppSession
# ======================================================================


class AppSession:
    def __init__(self, model_override: str | None = None):
        self.agent, self.cfg, self.current_model, self.memory_store, self.skills = (
            _build_runtime(model_override)
        )

    def refresh_model(self, model_name: str) -> None:
        self.agent, self.cfg, self.current_model, self.memory_store, self.skills = (
            _build_runtime(model_name)
        )


# ======================================================================
# CLI 入口
# ======================================================================


@click.group()
def main() -> None:
    """AuraDerma CLI."""


@main.command()
@click.option("--model", type=str, default=None, help="Override model name for this session")
@click.option("--user-id", default="default-user")
def chat(model: str | None, user_id: str) -> None:
    """启动交互式聊天会话（跨平台支持）。"""
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

        answer = session.agent.answer(
            AgentContext(user_id=user_id, question=cmd, memory=memory),
            memory_store=session.memory_store,
            skill_manager=session.skills,
        )
        click.echo("\n[assistant]\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")
        memory.short_term.append(
            session.agent.policy.classify(f"本轮对话：{cmd}", user_id=user_id)
        )
        session.agent.finalize_turn(
            user_id=user_id,
            dialog_text="\n\n".join(turn_log[-4:]),
            memory_store=session.memory_store,
            memory_bundle=memory,
        )


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
        answer = session.agent.answer(
            AgentContext(user_id=user_id, question=cmd, memory=memory),
            memory_store=session.memory_store,
            skill_manager=session.skills,
        )
        click.echo("\nassistant:\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")
        memory.short_term.append(
            session.agent.policy.classify(f"本轮对话：{cmd}", user_id=user_id)
        )
        session.agent.finalize_turn(
            user_id=user_id,
            dialog_text="\n\n".join(turn_log[-4:]),
            memory_store=session.memory_store,
            memory_bundle=memory,
        )


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
    from llm import LocalEmbedder

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
