from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SkillSpec:
    name: str
    summary: str
    body: str
    path: Path


class SkillsRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_skills(self) -> list[SkillSpec]:
        skills: list[SkillSpec] = []
        for skill_dir in sorted(self.root.iterdir()):
            if not skill_dir.is_dir():
                continue
            summary_path = skill_dir / "summary.md"
            body_path = skill_dir / "skill.md"
            if not summary_path.exists():
                continue
            skills.append(
                SkillSpec(
                    name=skill_dir.name,
                    summary=summary_path.read_text(encoding="utf-8").strip(),
                    body=body_path.read_text(encoding="utf-8").strip() if body_path.exists() else "",
                    path=skill_dir,
                )
            )
        return skills

    def as_summary_block(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "无可用 skills"
        lines = []
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.summary}")
        return "\n".join(lines)

    def get(self, name: str) -> SkillSpec | None:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        return None

    def body_block(self, names: list[str]) -> str:
        selected = []
        for name in names:
            skill = self.get(name)
            if skill and skill.body:
                selected.append(f"## {skill.name}\n{skill.body}")
        return "\n\n".join(selected)
