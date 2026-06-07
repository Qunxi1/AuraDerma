from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click

from core import get_logger

log = get_logger("auraderma.cli")


@dataclass(slots=True)
class CommandSpec:
    name: str
    description: str
    usage: str


COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec("/help", "show command help", "/help"),
    CommandSpec("/model", "switch or inspect model", "/model [name]"),
    CommandSpec("/models", "list model options", "/models"),
    CommandSpec("/search-engine", "switch web search provider", "/search-engine [name]"),
    CommandSpec("/search-config", "set api key for a search provider", "/search-config <name> <key>"),
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


def handle_command(cmd: str, session, user_id: str, memory) -> str:
    """处理斜杠命令，返回状态标记：'quit', 'rerender', 'switched'。"""
    if cmd == "/help":
        click.echo(help_text())
        return "rerender"
    if cmd == "/models":
        click.echo("available models:\n" + "\n".join(model_lines()))
        click.echo("tip: run `/model <name>` and tab-complete a choice")
        return "rerender"
    if cmd.startswith("/model"):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 1:
            click.echo(f"current model: {session.current_model}")
            click.echo("available models:\n" + "\n".join(model_lines()))
            return "rerender"
        chosen = parts[1].strip()
        if chosen not in MODEL_CHOICES:
            click.echo("unknown model; available options:\n" + "\n".join(model_lines()))
            return "rerender"
        session.refresh_model(chosen)
        click.echo(f"model switched to {session.current_model}")
        return "switched"
    if cmd == "/search-engine":
        _show_search_engines(session)
        return "rerender"
    if cmd.startswith("/search-engine"):
        parts = cmd.split(maxsplit=1)
        if len(parts) == 1:
            _show_search_engines(session)
            return "rerender"
        chosen = parts[1].strip().lower()
        try:
            session.agent.web.set_provider(chosen)
            click.echo(f"搜索提供方已切换为：{session.agent.web.current_provider_label}")
        except ValueError as e:
            click.echo(str(e))
        return "rerender"
    if cmd.startswith("/search-config"):
        return _handle_search_config(cmd, session)
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


def help_text() -> str:
    lines = ["commands:"]
    for spec in COMMAND_SPECS:
        lines.append(f"  {spec.usage:<16} {spec.description}")
    lines.append("")
    lines.append("autocomplete: type `/` and press Tab, or type `/model ` for model suggestions")
    lines.append("")
    lines.append("model picker:")
    lines.extend(f"  {line}" for line in model_lines())
    return "\n".join(lines)


def banner(model: str) -> str:
    return (
        "╭────────────────────────────────────────────╮\n"
        f"│  AuraDerma  -  model: {model:<22}│\n"
        "│  /help for commands - Tab for suggestions  │\n"
        "╰────────────────────────────────────────────╯"
    )


def model_lines() -> list[str]:
    return [f"{idx + 1}. {name}" for idx, name in enumerate(MODEL_CHOICES)]


def _apply_memory_to_bundle(bundle, item) -> None:
    if item.scope.value == "profile":
        bundle.profile.append(item)
    elif item.scope.value == "short_term":
        bundle.short_term.append(item)
    elif item.scope.value == "long_term":
        bundle.long_term.append(item)
    else:
        bundle.case_notes.append(item)


def _show_search_engines(session) -> None:
    from web_search import SEARCH_PROVIDERS

    cfg_path = Path.home() / ".auraderma" / "search_config.json"
    click.echo(f"当前搜索提供方：{session.agent.web.current_provider_label}")
    click.echo("")
    click.echo("可用搜索提供方：")
    max_key = max(len(k) for k in SEARCH_PROVIDERS) + 2
    for key, info in SEARCH_PROVIDERS.items():
        marker = " *" if key == session.agent.web.provider else ""
        needs = "需 API Key" if info["needs_api_key"] else "无需 Key"
        click.echo(f"  {key:<{max_key}} {info['label']}{marker}")
    click.echo("")
    click.echo("切换命令：/search-engine <名称>")
    click.echo("示例：/search-engine tavily")
    click.echo("")
    click.echo(f"所有 API Key 统一在 {cfg_path} 中管理，")
    click.echo("可用 /search-config <名称> <Key> 快速配置。")
    click.echo("也可通过同名环境变量（AURADERMA_*_API_KEY）直接覆盖。")


def _handle_search_config(cmd: str, session) -> str:
    from search_config import (
        PROVIDERS_NO_CONFIG,
        PROVIDER_TO_ENDPOINT_FIELD,
        save_api_key_for_provider,
    )

    parts = cmd.split(maxsplit=2)
    if len(parts) < 3:
        click.echo("用法：/search-config <提供方名称> <API Key>")
        click.echo("示例：/search-config tavily tvly-xxxxx")
        click.echo("")
        click.echo("支持的提供方：")
        _print_supported_for_config()
        return "rerender"

    provider_name = parts[1].strip().lower()
    api_key = parts[2].strip()

    if provider_name == "searxng":
        from search_config import save_search_config
        save_search_config(searxngEndpoint=api_key)
        click.echo(f"SearXNG 端点已保存为：{api_key}")
        return "rerender"

    if provider_name in PROVIDERS_NO_CONFIG:
        click.echo(f"'{provider_name}' 不需要 API Key，可直接使用 /search-engine {provider_name} 切换")
        return "rerender"

    err = save_api_key_for_provider(provider_name, api_key)
    if err:
        click.echo(err)
        click.echo("")
        _print_supported_for_config()
    else:
        click.echo(f"'{provider_name}' 的 API Key 已保存到配置文件")
        click.echo(f"可用 /search-engine {provider_name} 切换使用")
    return "rerender"


def _print_supported_for_config() -> None:
    from search_config import PROVIDER_TO_FIELD, PROVIDER_TO_ENDPOINT_FIELD, PROVIDERS_NO_CONFIG

    for provider, field in PROVIDER_TO_FIELD.items():
        click.echo(f"  {provider:<12} → 配置字段: {field}")
    for provider, field in PROVIDER_TO_ENDPOINT_FIELD.items():
        click.echo(f"  {provider:<12} → 配置字段: {field}（URL 地址，非 Key）")
    click.echo("")
    no_key_list = "、".join(sorted(PROVIDERS_NO_CONFIG))
    click.echo(f"无需配置（不需 Key）：{no_key_list}")
