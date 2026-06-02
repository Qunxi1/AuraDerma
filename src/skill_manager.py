from __future__ import annotations

import importlib.util
from pathlib import Path

from skills_registry import SkillsRegistry
from web_search import WebSearchClient

# Load skill implementations from skills/<skill_name>/skill.py
_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _load_skill(skill_name: str):
    """Load a skill module from skills/<skill_name>/skill.py via file path."""
    module_path = _SKILLS_DIR / skill_name / "skill.py"
    if not module_path.exists():
        raise ImportError(f"Skill module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"skills.{skill_name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ws_mod = _load_skill("web_search")
WebSearchSkill = _ws_mod.WebSearchSkill


class SkillManager:
    def __init__(self, skills_root: Path, web_client: WebSearchClient) -> None:
        self.registry = SkillsRegistry(skills_root)
        self.web_search = WebSearchSkill(web_client)

    def registry_summary(self) -> str:
        return self.registry.as_summary_block()

    def registry_body(self, names: list[str]) -> str:
        return self.registry.body_block(names)

    def web_search_payload(self, query: str, top_k: int = 5) -> dict:
        return self.web_search.to_tool_payload(query=query, top_k=top_k)
