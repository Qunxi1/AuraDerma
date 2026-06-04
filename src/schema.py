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


def new_id() -> str:
    return str(uuid.uuid4())
