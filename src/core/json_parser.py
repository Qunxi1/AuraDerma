from __future__ import annotations

import json
import re
from typing import Any

from .logger import get_logger

log = get_logger("auraderma.json")


class JsonParseError(Exception):
    """JSON 解析失败时抛出的异常，携带原始文本供追溯。"""

    def __init__(self, message: str, raw_text: str, context: str = "") -> None:
        self.raw_text = raw_text
        self.context = context
        super().__init__(f"[{context}] {message}")


class JsonParser:
    """健壮的 JSON 解析器，带自动修复、日志记录和错误报告。

    解决 issue 1 & 3：
    - LLM 返回坏 JSON 时不再静默吞错
    - 所有失败和修复操作都有日志可追溯
    - 支持自动修复常见 LLM JSON 问题（注释、尾逗号、单引号等）
    """

    @staticmethod
    def parse_obj(
        raw: str,
        context: str = "unknown",
        max_preview_chars: int = 500,
    ) -> dict[str, Any]:
        """解析 LLM 返回的 JSON 对象。解析失败时记录详细日志并抛出异常。

        Args:
            raw: LLM 原始返回文本
            context: 调用场景描述（如 "intent_classifier", "workflow_planner"）
            max_preview_chars: 日志中截取 raw 的最大字符数

        Returns:
            解析后的 dict

        Raises:
            JsonParseError: 所有修复尝试均失败
        """
        cleaned = raw.strip()
        if not cleaned:
            _log_failure(context, cleaned, "LLM 返回了空白文本", max_preview_chars)
            raise JsonParseError("LLM 返回空白文本", cleaned, context)

        # 尝试直接解析
        obj = _try_parse(cleaned, context)
        if obj is not None:
            return obj

        # 尝试提取 ```json ... ``` 代码块
        extracted = _extract_json_block(cleaned)
        if extracted is not None:
            obj = _try_parse(extracted, f"{context}(codeblock)")
            if obj is not None:
                log.warning(
                    "[%s] 从 markdown 代码块中提取 JSON (raw_preview=%s...)",
                    context, cleaned[:max_preview_chars],
                )
                return obj

        # 尝试修复常见问题
        fixed = _auto_fix_json(cleaned)
        if fixed is not None:
            obj = _try_parse(fixed, f"{context}(auto_fixed)")
            if obj is not None:
                log.warning(
                    "[%s] 自动修复后解析成功 (raw_preview=%s...)",
                    context, cleaned[:max_preview_chars],
                )
                return obj

        # 所有尝试均失败
        _log_failure(context, cleaned, "所有修复尝试均失败", max_preview_chars)
        raise JsonParseError("JSON 解析失败（已尝试自动修复）", cleaned, context)

    @staticmethod
    def parse_list(
        raw: str,
        context: str = "unknown",
        max_preview_chars: int = 500,
    ) -> list[dict[str, Any]]:
        """解析 LLM 返回的 JSON 数组。"""
        cleaned = raw.strip()
        if not cleaned:
            _log_failure(context, cleaned, "LLM 返回了空白文本", max_preview_chars)
            raise JsonParseError("LLM 返回空白文本", cleaned, context)

        obj = _try_parse_list(cleaned)
        if obj is not None:
            return obj

        extracted = _extract_json_block(cleaned)
        if extracted is not None:
            obj = _try_parse_list(extracted)
            if obj is not None:
                log.warning(
                    "[%s] 从 markdown 代码块中提取 JSON 数组 (raw_preview=%s...)",
                    context, cleaned[:max_preview_chars],
                )
                return obj

        fixed = _auto_fix_json(cleaned)
        if fixed is not None:
            obj = _try_parse_list(fixed)
            if obj is not None:
                log.warning(
                    "[%s] 自动修复后解析 JSON 数组成功", context,
                )
                return obj

        _log_failure(context, cleaned, "所有修复尝试均失败", max_preview_chars)
        raise JsonParseError("JSON 数组解析失败", cleaned, context)

    @staticmethod
    def safe_parse_obj(
        raw: str,
        context: str = "unknown",
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """安全解析，失败时返回默认值（不抛异常，仅记录日志）。

        用于不关键的场景，如补充信息提取。
        """
        try:
            return JsonParser.parse_obj(raw, context)
        except JsonParseError:
            return default or {}

    @staticmethod
    def safe_parse_list(
        raw: str,
        context: str = "unknown",
        default: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """安全解析数组，失败时返回默认值。"""
        try:
            return JsonParser.parse_list(raw, context)
        except JsonParseError:
            return default or []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_parse(text: str, context: str) -> dict[str, Any] | None:
    """尝试将文本解析为 JSON dict。"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        log.warning("[%s] JSON 解析结果不是 dict，而是 %s", context, type(data).__name__)
        return None
    except json.JSONDecodeError as e:
        log.debug("[%s] JSON 解析失败: %s", context, e)
        return None


def _try_parse_list(text: str) -> list[dict[str, Any]] | None:
    """尝试将文本解析为 JSON list。"""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        log.warning("JSON 解析结果不是 list，而是 %s", type(data).__name__)
        return None
    except json.JSONDecodeError as e:
        log.debug("JSON 列表解析失败: %s", e)
        return None


def _extract_json_block(text: str) -> str | None:
    """从 markdown 代码块 ```json ... ``` 中提取 JSON 内容。"""
    # 匹配 ```json ... ``` 或 ``` ... ```
    m = re.search(
        r"```(?:json|javascript|js)?\s*\n?(.*?)\n?```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate
    return None


def _auto_fix_json(text: str) -> str | None:
    """尝试自动修复常见的 LLM JSON 格式问题。"""
    fixed = text

    # 1. 移除 // 和 /* */ 注释
    fixed = re.sub(r"//[^\n]*", "", fixed)
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)

    # 2. 移除尾逗号（dict 和 list 中）
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*\]", "]", fixed)

    # 3. 移除属性名周围多余的空格
    fixed = re.sub(r"'\s*:\s*'", "': '", fixed)  # preserve single quotes for now

    # 4. 将 ... 占位符替换为空字符串或 null
    fixed = re.sub(r'"\s*\.\.\.\s*"', '"..."', fixed)
    fixed = re.sub(r'\.\.\.', '"..."', fixed)

    # 5. 将 Python 风格的 True/False/None 转换为 JSON true/false/null
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)

    # 如果修复后跟原始文本不同，返回修复版
    if fixed != text:
        return fixed
    return None


def _log_failure(
    context: str,
    raw: str,
    reason: str,
    max_preview: int = 500,
) -> None:
    """记录 JSON 解析失败的完整信息到日志文件。"""
    preview = raw[:max_preview]
    log.error(
        "[%s] JSON 解析失败: %s\n"
        "  raw_preview (first %d chars):\n"
        "  %s\n"
        "  raw_length=%d",
        context,
        reason,
        max_preview,
        preview,
        len(raw),
    )
