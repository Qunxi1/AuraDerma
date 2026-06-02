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
        }


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
