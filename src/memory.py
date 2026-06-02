from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json

from schema import MemoryItem, MemoryScope, new_id


@dataclass(slots=True)
class MemoryBundle:
    profile: list[MemoryItem] = field(default_factory=list)
    short_term: list[MemoryItem] = field(default_factory=list)
    long_term: list[MemoryItem] = field(default_factory=list)
    case_notes: list[MemoryItem] = field(default_factory=list)

    def all_items(self) -> list[MemoryItem]:
        return [*self.profile, *self.short_term, *self.long_term, *self.case_notes]

    def counts(self) -> dict[str, int]:
        return {
            "profile": len(self.profile),
            "short_term": len(self.short_term),
            "long_term": len(self.long_term),
            "case_notes": len(self.case_notes),
        }

    def summary_lines(self, max_items_per_scope: int = 5) -> list[str]:
        lines: list[str] = []
        for scope_name, items in [
            ("profile", self.profile),
            ("short_term", self.short_term),
            ("long_term", self.long_term),
            ("case_notes", self.case_notes),
        ]:
            for item in items[:max_items_per_scope]:
                lines.append(f"[{scope_name}] {item.summary}")
        return lines


class MemoryPolicy:
    """Hybrid memory policy for AuraDerma."""

    short_term_turns = 12
    long_term_default_ttl_days = 365

    def classify(self, text: str, user_id: str, source_doc_id: str | None = None) -> MemoryItem:
        normalized = text.strip()
        scope = MemoryScope.LONG_TERM
        ttl_days: int | None = self.long_term_default_ttl_days
        tags: list[str] = []

        if any(key in normalized for key in ["肤质", "过敏", "敏感", "油皮", "干皮", "混油", "痘肌"]):
            scope = MemoryScope.PROFILE
            ttl_days = None
            tags.append("profile")
        elif any(key in normalized for key in ["本轮", "刚刚", "最近", "今天", "这次对话"]):
            scope = MemoryScope.SHORT_TERM
            ttl_days = 7
            tags.append("dialogue")
        elif source_doc_id:
            scope = MemoryScope.CASE
            ttl_days = 180
            tags.append("report")

        return MemoryItem(
            memory_id=new_id(),
            scope=scope,
            user_id=user_id,
            text=normalized,
            summary=normalized[:120],
            confidence=0.7,
            ttl_days=ttl_days,
            tags=tags,
            source_doc_id=source_doc_id,
        )

    def should_promote(self, item: MemoryItem) -> bool:
        return item.scope in {MemoryScope.PROFILE, MemoryScope.LONG_TERM}

    def created_at(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def expires_at(self, item: MemoryItem) -> str | None:
        if item.ttl_days is None:
            return None
        return (datetime.utcnow() + timedelta(days=item.ttl_days)).isoformat(timespec="seconds") + "Z"


class MemoryStore:
    """File-per-memory storage with per-file index and recall helpers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.memories_dir = self.root / "memories"
        self.index_dir = self.root / "indexes"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def append(self, item: MemoryItem) -> Path:
        mem_path = self._memory_path(item)
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(json.dumps(self._serialize(item), ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_index(item, mem_path)
        return mem_path

    def load(self, user_id: str, limit: int = 200) -> MemoryBundle:
        bundle = MemoryBundle()
        for item in self._load_all_items(user_id=user_id, limit=limit):
            if item.scope == MemoryScope.PROFILE:
                bundle.profile.append(item)
            elif item.scope == MemoryScope.SHORT_TERM:
                bundle.short_term.append(item)
            elif item.scope == MemoryScope.LONG_TERM:
                bundle.long_term.append(item)
            else:
                bundle.case_notes.append(item)
        return bundle

    def load_index(self, user_id: str, limit: int = 30, scopes: set[str] | None = None) -> list[str]:
        lines: list[str] = []
        for index_file in sorted(self.index_dir.rglob("*.json")):
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if data.get("user_id") != user_id:
                continue
            if scopes and data.get("scope") not in scopes:
                continue
            lines.append(f"[{data.get('scope')}] {data.get('summary', '')}")
        return lines[-limit:]

    def scan_index_records(self, user_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index_file in sorted(self.index_dir.rglob("*.json")):
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if data.get("user_id") == user_id:
                records.append(data)
        return records

    def recall_index_lines(self, user_id: str, query: str, limit: int = 6) -> list[str]:
        records = self.scan_index_records(user_id)
        if not records:
            return []
        keywords = [k for k in _tokenize(query) if len(k) > 1]
        scored: list[tuple[int, dict[str, Any]]] = []
        for rec in records:
            hay = " ".join([rec.get("summary", ""), " ".join(rec.get("tags", [])), rec.get("scope", "")])
            score = sum(1 for k in keywords if k in hay)
            scored.append((score, rec))
        scored.sort(key=lambda x: (x[0], x[1].get("created_at", "")), reverse=True)
        selected = [rec for score, rec in scored if score > 0][:limit]
        if not selected:
            selected = [rec for _, rec in scored[:limit]]
        return [f"[{rec.get('scope')}] {rec.get('summary', '')}" for rec in selected]

    def read_relevant_memory_texts(self, user_id: str, memory_ids: list[str], query: str, limit: int = 4) -> list[str]:
        if not memory_ids:
            return []
        query_terms = [t for t in _tokenize(query) if len(t) > 1]
        snippets: list[str] = []
        for mem_file in sorted(self.memories_dir.rglob("*.json")):
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            if data.get("user_id") != user_id:
                continue
            if data.get("memory_id") not in memory_ids:
                continue
            text = str(data.get("text", "")).strip()
            if not text:
                continue
            snippet = _extract_relevant_snippet(text, query_terms)
            snippets.append(f"[{data.get('scope')}] {snippet}")
            if len(snippets) >= limit:
                break
        return snippets

    def rebuild_indexes(self) -> None:
        for mem_file in self.memories_dir.rglob("*.json"):
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            item = self._deserialize(data)
            self._write_index(item, mem_file, overwrite=True)

    def _memory_path(self, item: MemoryItem) -> Path:
        return self.memories_dir / item.user_id / item.scope.value / f"{item.memory_id}.json"

    def _index_path(self, item: MemoryItem) -> Path:
        return self.index_dir / item.user_id / item.scope.value / f"{item.memory_id}.json"

    def _write_index(self, item: MemoryItem, mem_path: Path, overwrite: bool = True) -> None:
        index_path = self._index_path(item)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory_id": item.memory_id,
            "user_id": item.user_id,
            "scope": item.scope.value,
            "summary": item.summary,
            "tags": item.tags,
            "source_doc_id": item.source_doc_id,
            "created_at": self.created_at(),
            "expires_at": self.expires_at(item),
            "memory_file": str(mem_path),
        }
        if overwrite or not index_path.exists():
            index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_all_items(self, user_id: str, limit: int = 200) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for mem_file in sorted(self.memories_dir.rglob("*.json")):
            if len(items) >= limit:
                break
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            if data.get("user_id") != user_id:
                continue
            items.append(self._deserialize(data))
        return items

    def _serialize(self, item: MemoryItem) -> dict[str, Any]:
        return item.to_payload() | {
            "created_at": self.created_at(),
            "expires_at": self.expires_at(item),
        }

    def _deserialize(self, data: dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            memory_id=data["memory_id"],
            scope=MemoryScope(data["scope"]),
            user_id=data["user_id"],
            text=data["text"],
            summary=data.get("summary", data["text"][:120]),
            confidence=float(data.get("confidence", 0.5)),
            ttl_days=data.get("ttl_days"),
            tags=list(data.get("tags", [])),
            source_doc_id=data.get("source_doc_id"),
        )

    def created_at(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def expires_at(self, item: MemoryItem) -> str | None:
        if item.ttl_days is None:
            return None
        return (datetime.utcnow() + timedelta(days=item.ttl_days)).isoformat(timespec="seconds") + "Z"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    buf = []
    for ch in text:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _extract_relevant_snippet(text: str, query_terms: list[str], max_len: int = 240) -> str:
    if not query_terms:
        return text[:max_len]
    lowered = text.lower()
    best_pos = None
    for term in query_terms:
        pos = lowered.find(term.lower())
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
    if best_pos is None:
        return text[:max_len]
    start = max(0, best_pos - 60)
    end = min(len(text), best_pos + max_len - 60)
    return text[start:end]
