from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent

from agent import IntentResult, WorkflowPlan
from core import JsonParser, get_logger
from prompts import SYSTEM_PROMPT, WORKFLOW_PLANNER_PROMPT

log = get_logger("auraderma.workflow")

_SKINCARE_PROCESSES = frozenset({
    "product_search", "skincare_analysis", "regimen_planning", "memory_lookup",
})


class WorkflowService:
    """工作流规划服务。

    根据用户问题和意图分类结果，决定需要执行哪些处理流程。
    """

    def __init__(self, llm):
        self._llm = llm

    def plan(self, question: str, intent: IntentResult) -> WorkflowPlan:
        """规划工作流。

        Args:
            question: 用户问题
            intent: 意图分类结果

        Returns:
            WorkflowPlan: 工作流计划
        """
        log.info(
            "规划工作流: intent=%s goal=%s",
            intent.intent, intent.goal,
        )

        prompt = dedent(f"""
            用户问题:
            {question}

            意图分类结果:
            - intent: {intent.intent}
            - goal: {intent.goal}
            - skincare_related: {intent.is_skincare_related}
            - reasoning: {intent.reasoning}

            请规划要执行的流程，返回 JSON。
        """).strip()

        raw = self._llm.chat(SYSTEM_PROMPT, f"{WORKFLOW_PLANNER_PROMPT}\n\n{prompt}")
        obj = JsonParser.parse_obj(raw, context="workflow_planner")
        processes = [str(p) for p in obj.get("processes", [])]

        # Safety net: general intent must have general_chat
        if intent.intent == "general" and "general_chat" not in processes:
            processes.insert(0, "general_chat")

        # Safety net: general intent should never have skincare processes
        if intent.intent == "general":
            processes = [p for p in processes if p not in _SKINCARE_PROCESSES]
            if "general_chat" not in processes:
                processes.insert(0, "general_chat")

        plan = WorkflowPlan(
            processes=processes,
            rationale=str(obj.get("rationale", "")),
            needs_product_search=bool(obj.get("needs_product_search", False)),
            needs_skincare_advice=bool(
                obj.get("needs_skincare_advice", "skincare_analysis" in processes)
            ),
        )
        log.info("工作流规划结果: processes=%s", plan.processes)
        return plan
