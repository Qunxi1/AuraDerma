from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
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


@dataclass(slots=True)
class CommandSpec:
    name: str
    description: str
    usage: str


COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec("/help", "show command help", "/help"),
    CommandSpec("/model", "switch or inspect model", "/model [name]"),
    CommandSpec("/models", "list model options", "/models"),
    CommandSpec("/memory", "show memory summary", "/memory"),
    CommandSpec("/skills", "show available skills", "/skills"),
    CommandSpec("/save", "save a memory note", "/save <note>"),
    CommandSpec("/quit", "quit the chat", "/quit"),
]

MODEL_CHOICES = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-reasoner",
    "qwen3.6-plus",
    "kimi-k2.5",
    "glm-5",
    "custom-openai-compatible",
]

def _aris_readline(
    model_name: str,
    command_specs: list[CommandSpec],
    model_names: list[str],
    history: list[str],
) -> str | None:
    """ARIS-style readline: render prompt + input + completion dropdown inline,
    using ANSI escape codes (no floating popup).

    Returns entered string, or None on exit.
    """
    try:
        import msvcrt
    except ImportError:
        raise NotImplementedError("_aris_readline requires Windows msvcrt")


    buf: list[str] = []
    cursor = 0
    sel = 0
    history_idx: int | None = None
    saved_buf: list[str] | None = None

    # Precompute prompt length (visible width, no ANSI codes)
    prompt_str = f"\x1b[36m\x1b[1mAuraDerma \x1b[34m[{model_name}] \x1b[33m\x1b[1m> \x1b[0m"
    _prompt_visible_len = 15 + len(model_name)  # "AuraDerma " + "[" + name + "] " + "› "

    def _compute_matches(line: str) -> list[tuple[str, str]]:
        """Return list of (name, description) that match the current input."""
        if line.startswith("/model ") and len(line) > 7:
            partial = line[7:]
            out = []
            for n in model_names:
                it = iter(n)
                if all(c in it for c in partial):
                    out.append((n, "model"))
            return out
        if not line.startswith("/"):
            return []
        text_lower = line.lower()
        out = []
        for sp in command_specs:
            it = iter(sp.name.lower())
            if all(c in it for c in text_lower):
                out.append((sp.name, sp.description))
        return out

    def _render() -> None:
        nonlocal sel
        line = "".join(buf)
        matches = _compute_matches(line)

        # Clear from start of prompt row to end of screen, then redraw
        sys.stdout.write("\r\x1b[J")

        # ── Draw prompt + input ──
        sys.stdout.write(prompt_str)
        sys.stdout.write(line)

        # ── Draw dropdown below ──
        if matches:
            max_name = max(len(m[0]) for m in matches)
            name_col = min(max(max_name, 12), 36) + 2
            if sel >= len(matches):
                sel = len(matches) - 1

            # Separator line
            sys.stdout.write("\r\n")
            sys.stdout.write(f"\x1b[2m{'─' * 60}\x1b[0m")
            row_count = 2

            for idx, (name, desc) in enumerate(matches):
                sys.stdout.write("\r\n")
                row_count += 1
                if idx == sel:
                    sys.stdout.write(f"\x1b[1;34m{name}\x1b[0m")
                    sys.stdout.write(" " * (name_col - len(name)))
                    sys.stdout.write(f"\x1b[1;33m{desc}\x1b[0m")
                else:
                    sys.stdout.write(name)
                    sys.stdout.write(" " * (name_col - len(name)))
                    sys.stdout.write(f"\x1b[33m{desc}\x1b[0m")

            # Move cursor back to the prompt line for typing
            sys.stdout.write(f"\x1b[{row_count - 1}A")

        # Position cursor within input text (<ESC>[G is 1-indexed)
        col = _prompt_visible_len + 1 + cursor
        sys.stdout.write(f"\x1b[{col}G")
        sys.stdout.flush()

    _render()

    while True:
        ch = msvcrt.getwch()

        # ── Arrow / function keys (prefixed with \xe0) ──
        if ch == "\xe0":
            ch2 = msvcrt.getwch()
            if ch2 == "H":  # Up
                if matches_list := _compute_matches("".join(buf)):
                    if sel > 0:
                        sel -= 1
                elif history:
                    # History
                    if history_idx is None:
                        saved_buf = buf[:]
                        history_idx = len(history) - 1
                    elif history_idx > 0:
                        history_idx -= 1
                    buf = list(history[history_idx])
                    cursor = len(buf)
                    sel = 0
                _render()
            elif ch2 == "P":  # Down
                if matches_list := _compute_matches("".join(buf)):
                    if sel < len(_compute_matches("".join(buf))) - 1:
                        sel += 1
                elif history_idx is not None:
                    if history_idx < len(history) - 1:
                        history_idx += 1
                        buf = list(history[history_idx])
                    else:
                        history_idx = None
                        buf = saved_buf or []
                    cursor = len(buf)
                    sel = 0
                _render()
            elif ch2 == "K":  # Left
                if cursor > 0:
                    cursor -= 1
                    _render()
            elif ch2 == "M":  # Right
                if cursor < len(buf):
                    cursor += 1
                    _render()
            elif ch2 == "G":  # Home
                cursor = 0
                _render()
            elif ch2 == "O":  # End
                cursor = len(buf)
                _render()
            elif ch2 == "S":  # Delete
                if cursor < len(buf):
                    buf.pop(cursor)
                    sel = 0
                    _render()
            continue

        # ── Enter ──
        if ch == "\r":
            # Clear dropdown artifacts before committing
            sys.stdout.write("\r\x1b[J")
            sys.stdout.write(prompt_str + "".join(buf))
            sys.stdout.write("\n")
            sys.stdout.flush()
            result = "".join(buf)
            if result and result.strip():
                history.append(result.strip())
            return result

        # ── Backspace ──
        if ch in ("\b", "\x7f"):
            if cursor > 0:
                buf.pop(cursor - 1)
                cursor -= 1
                sel = 0
                _render()
            continue

        # ── Tab: accept selected completion ──
        if ch == "\t":
            full = "".join(buf)
            matches = _compute_matches(full)
            if matches and sel < len(matches):
                buf = list(matches[sel][0])
                cursor = len(buf)
                sel = 0
                _render()
            continue

        # ── Escape: close dropdown ──
        if ch == "\x1b":
            sel = 0
            _render()
            continue

        # ── Ctrl+C / Ctrl+D: exit ──
        if ch in ("\x03", "\x04"):
            sys.stdout.write("\r\x1b[J\n")
            sys.stdout.flush()
            return None

        # ── Regular character ──
        buf.insert(cursor, ch)
        cursor += 1
        sel = 0
        history_idx = None
        saved_buf = None
        _render()




class AppSession:
    def __init__(self, model_override: str | None = None):
        self.agent, self.cfg, self.current_model, self.memory_store, self.skills = _build_runtime(model_override)

    def refresh_model(self, model_name: str) -> None:
        self.agent, self.cfg, self.current_model, self.memory_store, self.skills = _build_runtime(model_name)


def _load_env() -> None:
    load_dotenv()
    load_dotenv(".env.local", override=True)


def _build_runtime(model_override: str | None = None) -> tuple[SkincareAgent, AppConfig, str, MemoryStore, SkillManager]:
    _load_env()
    # Suppress Qdrant version mismatch warning (client 1.18 vs server 1.11)
    warnings.filterwarnings("ignore", message=".*incompatible with server version.*")
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


def _apply_memory_to_bundle(bundle: MemoryBundle, item) -> None:
    if item.scope.value == "profile":
        bundle.profile.append(item)
    elif item.scope.value == "short_term":
        bundle.short_term.append(item)
    elif item.scope.value == "long_term":
        bundle.long_term.append(item)
    else:
        bundle.case_notes.append(item)


def _command_lines() -> list[str]:
    return [f"{spec.name:<8} {spec.description}" for spec in COMMAND_SPECS]


def _model_lines() -> list[str]:
    return [f"{idx + 1}. {name}" for idx, name in enumerate(MODEL_CHOICES)]


@click.group()
def main() -> None:
    """AuraDerma CLI."""


@main.command()
@click.option("--model", type=str, default=None, help="Override model name for this session")
@click.option("--user-id", default="default-user")
def chat(model: str | None, user_id: str) -> None:
    session = AppSession(model)
    memory = session.memory_store.load(user_id)
    turn_log: list[str] = []
    click.echo(_banner(session.current_model))
    sys.stdout.flush()

    history: list[str] = []

    while True:
        try:
            raw = _aris_readline(
                session.current_model,
                COMMAND_SPECS,
                MODEL_CHOICES,
                history,
            )
        except NotImplementedError:
            _legacy_chat(session, user_id, memory, turn_log)
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
            handled = _handle_command(cmd, session, user_id, memory)
            if handled == "quit":
                break
            if handled == "rerender":
                continue
            if handled == "switched":
                click.echo(_banner(session.current_model))
            continue

        answer = session.agent.answer(AgentContext(user_id=user_id, question=cmd, memory=memory), memory_store=session.memory_store, skill_manager=session.skills)
        click.echo("\n[assistant]\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")
        memory.short_term.append(session.agent.policy.classify(f"本轮对话：{cmd}", user_id=user_id))
        session.agent.finalize_turn(user_id=user_id, dialog_text="\n\n".join(turn_log[-4:]), memory_store=session.memory_store, memory_bundle=memory)


def _handle_command(cmd: str, session: AppSession, user_id: str, memory: MemoryBundle) -> str:
    if cmd == "/help":
        click.echo(_help_text())
        return "rerender"
    if cmd == "/models":
        click.echo("available models:\n" + "\n".join(_model_lines()))
        click.echo("tip: run `/model <name>` and tab-complete a choice")
        return "rerender"
    if cmd.startswith("/model"):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 1:
            click.echo(f"current model: {session.current_model}")
            click.echo("available models:\n" + "\n".join(_model_lines()))
            return "rerender"
        chosen = parts[1].strip()
        if chosen not in MODEL_CHOICES:
            click.echo("unknown model; available options:\n" + "\n".join(_model_lines()))
            return "rerender"
        session.refresh_model(chosen)
        click.echo(f"model switched to {session.current_model}")
        return "switched"
    if cmd == "/memory":
        click.echo(json.dumps(memory.counts(), ensure_ascii=False, indent=2))
        click.echo("\n" + "\n".join(session.memory_store.load_index(user_id=user_id)))
        return "rerender"
    if cmd == "/skills":
        click.echo(session.skills.registry_summary())
        return "rerender"
    if cmd.startswith("/save"):
        note = cmd.removeprefix("/save").strip()
        if not note:
            click.echo("usage: /save <note>")
            return "rerender"
        item = session.agent.policy.classify(note, user_id=user_id)
        session.memory_store.append(item)
        _apply_memory_to_bundle(memory, item)
        click.echo(f"saved memory: {item.scope.value} -> {item.summary}")
        return "rerender"
    click.echo("unknown command; try /help")
    return "rerender"


def _help_text() -> str:
    lines = ["commands:"]
    for spec in COMMAND_SPECS:
        lines.append(f"  {spec.usage:<16} {spec.description}")
    lines.append("")
    lines.append("autocomplete: type `/` and press Tab, or type `/model ` for model suggestions")
    lines.append("")
    lines.append("model picker:")
    lines.extend(f"  {line}" for line in _model_lines())
    return "\n".join(lines)


def _banner(model: str) -> str:
    return (
        "╭────────────────────────────────────────────╮\n"
        f"│  AuraDerma  -  model: {model:<22}│\n"
        "│  /help for commands - Tab for suggestions  │\n"
        "╰────────────────────────────────────────────╯"
    )


def _legacy_chat(session: AppSession, user_id: str, memory: MemoryBundle, turn_log: list[str]) -> None:
    click.echo("msvcrt unavailable on this platform; falling back to simple input mode.")
    while True:
        raw = click.prompt(f"AuraDerma[{session.current_model}]", type=str)
        cmd = raw.strip()
        if cmd.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if cmd.startswith("/"):
            handled = _handle_command(cmd, session, user_id, memory)
            if handled == "quit":
                break
            continue
        answer = session.agent.answer(AgentContext(user_id=user_id, question=cmd, memory=memory), memory_store=session.memory_store, skill_manager=session.skills)
        click.echo("\nassistant:\n" + answer + "\n")
        turn_log.append(f"user: {cmd}\nassistant: {answer}")
        memory.short_term.append(session.agent.policy.classify(f"本轮对话：{cmd}", user_id=user_id))
        session.agent.finalize_turn(user_id=user_id, dialog_text="\n\n".join(turn_log[-4:]), memory_store=session.memory_store, memory_bundle=memory)


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
        "models": MODEL_CHOICES,
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
