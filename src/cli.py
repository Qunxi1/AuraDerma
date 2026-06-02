from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from agent import AgentContext, SkincareAgent
from config import AppConfig, load_config
from llm import LLMClient
from memory import MemoryBundle, MemoryPolicy, MemoryStore
from retrieval import Retriever
from skill_manager import SkillManager
from web_search import WebSearchClient


def _load_env() -> None:
    load_dotenv()
    load_dotenv(".env.local", override=True)


def _build_runtime(model_override: str | None = None) -> tuple[SkincareAgent, AppConfig, str, MemoryStore, SkillManager]:
    _load_env()
    cfg = load_config()
    qdrant = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    retriever = Retriever(qdrant, cfg.qdrant_collection_products, cfg.qdrant_collection_memory, cfg.qdrant_collection_docs)
    retriever.ensure_collections()
    llm = LLMClient(api_key=cfg.model_api_key, base_url=cfg.model_api_base, model=model_override or cfg.default_model)
    web = WebSearchClient(enabled=cfg.web_search_enabled)
    policy = MemoryPolicy()
    agent = SkincareAgent(llm=llm, retriever=retriever, web=web, policy=policy)
    store = MemoryStore(cfg.data_dir / "memory")
    skills = SkillManager(cfg.skills_dir, web)
    return agent, cfg, llm.model, store, skills


@click.group()
def main() -> None:
    """AuraDerma CLI."""


@main.command()
@click.option("--model", type=str, default=None, help="Override model name for this session")
@click.option("--user-id", default="default-user")
def chat(model: str | None, user_id: str) -> None:
    agent, _, current_model, memory_store, skills = _build_runtime(model)
    click.echo("AuraDerma CLI started. 输入 /help 查看命令，输入 exit 退出。")
    memory = memory_store.load(user_id)
    turn_log: list[str] = []
    while True:
        raw = click.prompt(f"AuraDerma[{current_model}]", type=str)
        cmd = raw.strip()
        if cmd.lower() in {"exit", "quit"}:
            break
        if cmd.startswith("/"):
            if cmd == "/help":
                click.echo("/help /model <name> /memory /skills /save <text> /exit")
                continue
            if cmd.startswith("/model"):
                parts = cmd.split(maxsplit=1)
                if len(parts) == 1:
                    click.echo(f"current model: {current_model}")
                else:
                    agent, _, current_model, memory_store, skills = _build_runtime(parts[1].strip())
                    click.echo(f"model switched to {current_model}")
                continue
            if cmd == "/memory":
                click.echo(json.dumps(memory.counts(), ensure_ascii=False, indent=2))
                click.echo("\n" + "\n".join(memory_store.load_index(user_id=user_id)))
                continue
            if cmd == "/skills":
                click.echo(skills.registry_summary())
                continue
            if cmd.startswith("/save"):
                note = cmd.removeprefix("/save").strip()
                item = agent.policy.classify(note, user_id=user_id)
                memory_store.append(item)
                _apply_memory_to_bundle(memory, item)
                click.echo(f"saved memory: {item.scope.value} -> {item.summary}")
                continue
            click.echo("unknown command")
            continue

        answer = agent.answer(AgentContext(user_id=user_id, question=cmd, memory=memory), memory_store=memory_store, skill_manager=skills)
        click.echo("\nassistant:\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")
        memory.short_term.append(agent.policy.classify(f"本轮对话：{cmd}", user_id=user_id))
        agent.finalize_turn(user_id=user_id, dialog_text="\n\n".join(turn_log[-4:]), memory_store=memory_store, memory_bundle=memory)


@main.command()
def init() -> None:
    _load_env()
    cfg = load_config()
    click.echo(json.dumps({
        "model_base": cfg.model_api_base,
        "default_model": cfg.default_model,
        "qdrant_url": cfg.qdrant_url,
        "skills_dir": str(cfg.skills_dir),
        "collections": [cfg.qdrant_collection_products, cfg.qdrant_collection_memory, cfg.qdrant_collection_docs],
    }, ensure_ascii=False, indent=2))


@main.command()
@click.argument("path", type=click.Path(path_type=Path, exists=True))
def ingest(path: Path) -> None:
    from ingest import ingest_document

    result = ingest_document(path)
    click.echo(json.dumps({
        "doc_id": result.doc_id,
        "path": str(result.path),
        "type": result.doc_type.value,
        "chunks": len(result.chunks),
    }, ensure_ascii=False, indent=2))


def _apply_memory_to_bundle(bundle: MemoryBundle, item) -> None:
    if item.scope.value == "profile":
        bundle.profile.append(item)
    elif item.scope.value == "short_term":
        bundle.short_term.append(item)
    elif item.scope.value == "long_term":
        bundle.long_term.append(item)
    else:
        bundle.case_notes.append(item)
