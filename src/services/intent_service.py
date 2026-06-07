from __future__ import annotations

from dataclasses import dataclass, field

from agent import IntentResult  # reuse existing dataclass
from core import JsonParser, get_logger
from prompts import INTENT_CLASSIFIER_PROMPT, SYSTEM_PROMPT

log = get_logger("auraderma.intent")


class IntentService:
    """意图分类服务。

    将 LLM 的意图分类能力封装为独立服务，便于测试和替换策略。
    """

    def __init__(self, llm):
        self._llm = llm

    def classify(self, question: str) -> IntentResult:
        """分析用户问题意图。

        Args:
            question: 用户输入的问题文本

        Returns:
            IntentResult: 意图分类结果

        Raises:
            JsonParseError: LLM 返回的 JSON 无法解析
        """
        log.info("分类意图: question_preview=%s...", question[:80])

        raw = self._llm.chat(
            SYSTEM_PROMPT,
            f"{INTENT_CLASSIFIER_PROMPT}\n\n用户问题：{question}",
        )
        obj = JsonParser.parse_obj(raw, context="intent_classifier")

        result = IntentResult(
            intent=obj.get("intent", "single"),
            goal=obj.get("goal", "护肤咨询"),
            has_explicit_category=bool(obj.get("has_explicit_category", False)),
            explicit_categories=[str(c) for c in obj.get("explicit_categories", [])],
            reasoning=str(obj.get("reasoning", "")),
            is_skincare_related=bool(obj.get("is_skincare_related", True)),
        )
        log.info(
            "意图分类结果: intent=%s goal=%s categories=%s",
            result.intent, result.goal, result.explicit_categories,
        )
        return result
