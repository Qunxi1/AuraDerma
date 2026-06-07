from __future__ import annotations

from textwrap import dedent

from agent import RegimenPlan, RegimenStep
from core import JsonParser, get_logger
from prompts import REGIMEN_PLANNER_PROMPT, SYSTEM_PROMPT

log = get_logger("auraderma.regimen")


class RegimenService:
    """护肤体系规划服务。

    根据用户目标规划完整的早晚护理流程。
    """

    _TIME_SLOTS = [
        ("morning", "morning_steps"),
        ("evening", "evening_steps"),
        ("periodic", "periodic_steps"),
    ]

    def __init__(self, llm):
        self._llm = llm

    def plan(self, question: str, goal: str) -> RegimenPlan:
        """规划护肤体系。

        Args:
            question: 用户问题
            goal: 护肤目标（如 "美白"、"祛痘"）

        Returns:
            RegimenPlan: 护肤体系计划
        """
        log.info("规划护肤体系: goal=%s", goal)

        prompt = dedent(f"""
            {REGIMEN_PLANNER_PROMPT}

            用户问题：
            {question}

            识别出的护肤目标：{goal}

            请规划护肤体系，返回 JSON。
        """).strip()

        raw = self._llm.chat(SYSTEM_PROMPT, prompt)
        obj = JsonParser.parse_obj(raw, context="regimen_planner")

        steps: list[RegimenStep] = []
        for time_slot, key in self._TIME_SLOTS:
            for step_data in obj.get(key, []):
                steps.append(RegimenStep(
                    category=str(step_data.get("category", "")),
                    purpose=str(step_data.get("purpose", "")),
                    search_query=str(step_data.get("search_query", "")),
                    time_of_day=time_slot,
                ))

        plan = RegimenPlan(
            goal=goal,
            goal_explanation=str(obj.get("goal_explanation", "")),
            steps=steps,
            must_have_categories=[str(c) for c in obj.get("must_have_categories", [])],
            avoid_ingredients=[str(i) for i in obj.get("avoid_ingredients", [])],
            notes=str(obj.get("notes", "")),
            category_priority=[str(c) for c in obj.get("category_priority", [])],
        )
        log.info("护肤体系规划完成: %d 个步骤", len(steps))
        return plan
