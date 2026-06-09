from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import uuid


class MemoryScope(str, Enum):
    PROFILE = "profile"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    CASE = "case"


class DocType(str, Enum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEB = "web"
    PRODUCT = "product"
    MEMORY = "memory"
    TREATMENT = "treatment"


@dataclass(slots=True)
class ProductRecord:
    product_id: str
    name: str
    brand: str
    category: str
    price_cny: float | None = None
    price_note: str | None = None
    ingredients: list[str] = field(default_factory=list)
    ingredient_ordered_text: str = ""
    skin_types: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    usage_notes: str = ""
    source: str = ""
    # --- 新增字段，来源于线上采集的丰富护肤品数据 ---
    efficacy: str = ""              # 功效完整描述
    core_efficacy: str = ""         # 核心功效
    faq: list[str] = field(default_factory=list)        # 常见问题 Q&A，每个元素为 "Q: ... A: ..."
    model_type: str = ""            # 型号/类别 (如 二类医疗器械、R型)
    series: str = ""                # 品牌内部系列
    net_content: str = ""           # 净含量/规格
    storage: str = ""               # 贮存说明
    usage_steps: str = ""           # 详细使用步骤
    warnings: str = ""              # 注意事项/警告
    search_text: str = ""           # 用于向量化的聚合搜索文本

    def to_payload(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "price_cny": self.price_cny,
            "price_note": self.price_note,
            "ingredients": self.ingredients,
            "ingredient_ordered_text": self.ingredient_ordered_text,
            "skin_types": self.skin_types,
            "concerns": self.concerns,
            "usage_notes": self.usage_notes,
            "source": self.source,
            "efficacy": self.efficacy,
            "core_efficacy": self.core_efficacy,
            "faq": self.faq,
            "model_type": self.model_type,
            "series": self.series,
            "net_content": self.net_content,
            "storage": self.storage,
            "usage_steps": self.usage_steps,
            "warnings": self.warnings,
        }

    def build_search_text(self) -> str:
        """构建用于向量检索的聚合搜索文本"""
        parts: list[str] = []
        if self.name:
            parts.append(f"产品名称: {self.name}")
        if self.brand:
            parts.append(f"品牌: {self.brand}")
        if self.series:
            parts.append(f"系列: {self.series}")
        if self.category:
            parts.append(f"类别: {self.category}")
        if self.model_type:
            parts.append(f"型号: {self.model_type}")
        if self.efficacy:
            parts.append(f"功效: {self.efficacy}")
        if self.core_efficacy:
            parts.append(f"核心功效: {self.core_efficacy}")
        if self.skin_types:
            parts.append(f"适用肤质: {'、'.join(self.skin_types)}")
        if self.concerns:
            parts.append(f"针对问题: {'、'.join(self.concerns)}")
        if self.ingredients:
            parts.append(f"核心成分: {'、'.join(self.ingredients[:30])}")
        if self.net_content:
            parts.append(f"规格: {self.net_content}")
        if self.price_note:
            parts.append(f"价格: {self.price_note}")
        if self.usage_notes:
            parts.append(f"使用说明: {self.usage_notes}")
        if self.usage_steps:
            parts.append(f"使用步骤: {self.usage_steps}")
        if self.storage:
            parts.append(f"贮存: {self.storage}")
        if self.warnings:
            parts.append(f"注意事项: {self.warnings}")
        if self.faq:
            parts.append(f"常见问题: {'; '.join(self.faq)}")
        return "\n".join(parts)


@dataclass(slots=True)
class MemoryItem:
    memory_id: str
    scope: MemoryScope
    user_id: str
    text: str
    summary: str
    confidence: float = 0.5
    ttl_days: int | None = None
    tags: list[str] = field(default_factory=list)
    source_doc_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope.value,
            "user_id": self.user_id,
            "text": self.text,
            "summary": self.summary,
            "confidence": self.confidence,
            "ttl_days": self.ttl_days,
            "tags": self.tags,
            "source_doc_id": self.source_doc_id,
        }


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_type: DocType
    text: str
    title: str = ""
    section: str = ""
    page: int | None = None
    chunk_index: int = 0
    source_uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type.value,
            "text": self.text,
            "title": self.title,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "source_uri": self.source_uri,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Turn:
    """单轮对话记录。"""
    turn_index: int
    user_question: str
    assistant_answer: str
    timestamp: str = ""
    memory_ids: list[str] = field(default_factory=list)

    def to_summary_line(self, include_answer: bool = True) -> str:
        summary = f"[第{self.turn_index}轮] 用户: {self.user_question[:80]}"
        if include_answer:
            summary += f"\n         助手: {self.assistant_answer[:120]}"
        return summary


@dataclass(slots=True)
class ConversationHistory:
    """对话历史管理 —— 类似 Reasonix 的 Append-Only Log。

    维护多轮对话，当上下文窗口使用率接近上限时，
    用 LLM 对早期轮次做结构化摘要，保留最近几轮全文。

    压缩时机由外部决定（基于 API 返回的 prompt_tokens 占比），
    ConversationHistory 只提供数据结构和文本生成。
    """
    turns: list[Turn] = field(default_factory=list)
    _compacted_up_to: int = 0          # 已压缩到第几轮
    _compaction_summary: str = ""      # 压缩摘要文本

    # Reasonix 风格的压缩比例常量
    SOFT_COMPACT_RATIO: float = 0.5    # 达到 50% 时预警（不压缩，保持缓存）
    COMPACT_RATIO: float = 0.75        # 达到 75% 时触发压缩
    FORCE_COMPACT_RATIO: float = 0.9   # 达到 90% 时强制压缩
    # 兜底线：未压缩轮次超过此数也触发压缩
    # 因为 window=1M 时按比例触发需要~1500轮，
    # 但太长 history 会让每次请求又重又慢
    MAX_UNCOMPACTED_TURNS: int = 60

    @property
    def latest_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recent_turns(self, n: int = 10) -> list[Turn]:
        """获取最近 n 轮对话（完整内容）。"""
        return self.turns[-n:] if len(self.turns) >= n else self.turns[:]

    def compactable_region(self, keep_recent: int = 10) -> list[Turn] | None:
        """获取可以压缩的早期轮次（不包括最近 keep_recent 轮）。"""
        if len(self.turns) <= keep_recent + 1:
            return None
        return self.turns[: -keep_recent]

    def mark_compacted(self, summary: str, up_to_turn: int) -> None:
        """标记压缩边界。"""
        self._compacted_up_to = up_to_turn
        self._compaction_summary = summary

    def estimate_tokens(self, max_verbose_turns: int = 10) -> int:
        """估算当前 history_block 的 token 数，用于预压缩判断。

        使用保守估算（偏大），与 LLMClient.estimate_tokens 一致：
        CJK ~1.5 chars/token，非 CJK ~3.5 chars/token。
        """
        block = self.history_block(max_verbose_turns)
        if not block:
            return 0
        chars = len(block)
        cjk_count = sum(1 for ch in block if '\u4e00' <= ch <= '\u9fff')
        non_cjk = chars - cjk_count
        return int(cjk_count * 1.5 + non_cjk / 3.5) + 8

    def should_compact(self, prompt_ratio: float | None) -> bool:
        """基于 prompt 占比或轮次阈值判断是否需要压缩。

        Args:
            prompt_ratio: 最近一次 API 调用的 prompt_tokens / context_window。
                          None 表示尚无用量数据。

        Returns:
            是否需要压缩
        """
        # ① 比例触发（大窗口模型如 1M 很少靠这个触发）
        if prompt_ratio is not None:
            if prompt_ratio >= self.FORCE_COMPACT_RATIO:
                return True
            if prompt_ratio >= self.COMPACT_RATIO and self.compactable_region() is not None:
                return True
        # ② 轮次兜底触发：未压缩轮次超过上限（防止首次压缩遥遥无期）
        if not self._compaction_summary and self.turn_count >= self.MAX_UNCOMPACTED_TURNS:
            region = self.compactable_region()
            return region is not None
        return False

    def history_block(self, max_verbose_turns: int = 10) -> str:
        """生成对话历史文本块，用于注入 LLM context。

        类似于 Reasonix 的 Compose：
        - 有压缩摘要时：摘要在前，最近几轮完整内容在后
        - 无压缩摘要时：展示全部轮次
        """
        parts: list[str] = []

        # ① 压缩摘要（如果存在）
        if self._compaction_summary:
            parts.append(f"【早期对话摘要】\n{self._compaction_summary}\n")

        # ② 对话记录
        if self._compaction_summary:
            display_turns = self.recent_turns(max_verbose_turns)
        else:
            display_turns = self.turns[:]  # 未压缩时展示全部

        if display_turns:
            label = "【最近对话记录】" if self._compaction_summary else "【对话历史】"
            lines = [label]
            for t in display_turns:
                lines.append(t.to_summary_line(include_answer=True))
            parts.append("\n".join(lines))

        if not parts:
            return ""
        return "\n\n".join(parts)


def new_id() -> str:
    return str(uuid.uuid4())
