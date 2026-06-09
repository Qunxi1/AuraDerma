from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json

from schema import ConversationHistory, MemoryItem, MemoryScope, Turn, new_id


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

    # ---------- 肤质自然语言模式库 ----------
    # 用于 classify() 的自然语言匹配
    _SKIN_TYPE_PATTERNS: list[str] = [
        # 明确肤质
        "油皮", "干皮", "混油", "混干", "中性皮", "敏感皮", "痘肌",
        "油性", "干性", "混合性", "中性", "敏感性",
        # 自然表达：T区
        "t区油", "t区出油", "t区偏油", "t区很油", "t区爱出油",
        "t区干", "t区偏干", "t区很干",
        # 自然表达：脸颊/两颊
        "脸颊干", "脸颊偏干", "脸颊很干", "两颊干", "两颊偏干",
        "脸颊油", "脸颊偏油", "两颊油", "两颊出油",
        # 自然表达：综合
        "外油内干", "内油外干", "又油又干",
        "额头油", "下巴油", "鼻子油",
        # 敏感/过敏
        "容易过敏", "容易泛红", "容易敏感", "皮肤薄", "屏障受损",
        "泛红", "红血丝", "刺痛", "发痒", "起皮", "脱皮",
        # 痘痘相关
        "爱长痘", "容易长痘", "反复长痘", "闭口", "粉刺", "痘痘",
    ]

    def classify(self, text: str, user_id: str, source_doc_id: str | None = None) -> MemoryItem:
        normalized = text.strip().lower()
        scope = MemoryScope.LONG_TERM
        ttl_days: int | None = self.long_term_default_ttl_days
        tags: list[str] = []

        if any(pattern in normalized for pattern in self._SKIN_TYPE_PATTERNS):
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

    # ------------------------------------------------------------------
    # 对话历史持久化（类似 Reasonix 的 Append-Only Log 持久层）
    # ------------------------------------------------------------------

    def _conversations_dir(self, user_id: str) -> Path:
        """对话历史存储目录。"""
        d = self.root / "conversations" / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_conversation_history(self, user_id: str, history: ConversationHistory) -> None:
        """持久化对话历史到 JSON 文件。"""
        if not history.turns:
            return
        data = {
            "compacted_up_to": history._compacted_up_to,
            "compaction_summary": history._compaction_summary,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_question": t.user_question,
                    "assistant_answer": t.assistant_answer,
                    "timestamp": t.timestamp,
                    "memory_ids": t.memory_ids,
                }
                for t in history.turns
            ],
        }
        path = self._conversations_dir(user_id) / "history.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_conversation_history(self, user_id: str) -> ConversationHistory:
        """从持久化文件加载对话历史。"""
        path = self._conversations_dir(user_id) / "history.json"
        if not path.exists():
            return ConversationHistory()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            history = ConversationHistory()
            history._compacted_up_to = data.get("compacted_up_to", 0)
            history._compaction_summary = data.get("compaction_summary", "")
            for t_data in data.get("turns", []):
                history.turns.append(Turn(
                    turn_index=t_data["turn_index"],
                    user_question=t_data["user_question"],
                    assistant_answer=t_data["assistant_answer"],
                    timestamp=t_data.get("timestamp", ""),
                    memory_ids=t_data.get("memory_ids", []),
                ))
            return history
        except (json.JSONDecodeError, KeyError):
            return ConversationHistory()

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


# ======================================================================
# 结构化肤质画像 —— 将零散的 profile 聚合为结构化摘要
# ======================================================================


@dataclass(slots=True)
class SkinProfile:
    skin_type: str | None = None          # 油皮/干皮/混油/混干/中性/敏感
    t_zone: str | None = None             # T区：油/干/正常
    cheeks: str | None = None             # 脸颊：油/干/正常
    sensitivity: str | None = None        # 敏感程度：敏感/一般/耐受
    acne_prone: bool | None = None        # 是否易长痘
    concerns: list[str] = field(default_factory=list)  # 护肤诉求列表
    allergies: list[str] = field(default_factory=list)  # 过敏/避开的成分
    climate: str | None = None            # 所在地气候
    raw_memories: list[str] = field(default_factory=list)  # 原始记忆文本

    def is_empty(self) -> bool:
        return not any([
            self.skin_type, self.t_zone, self.cheeks,
            self.sensitivity, self.acne_prone,
            self.concerns, self.allergies, self.climate,
        ])

    def to_formatted_block(self) -> str:
        """生成 LLM-friendly 的结构化肤质画像文本块。"""
        if self.is_empty():
            return ""
        lines = ["【用户肤质画像】"]
        if self.skin_type:
            lines.append(f"  肤质类型: {self.skin_type}")
        if self.t_zone:
            lines.append(f"  T区: {self.t_zone}")
        if self.cheeks:
            lines.append(f"  脸颊: {self.cheeks}")
        if self.sensitivity:
            lines.append(f"  敏感程度: {self.sensitivity}")
        if self.acne_prone is not None:
            lines.append(f"  是否易长痘: {'是' if self.acne_prone else '否'}")
        if self.concerns:
            lines.append(f"  护肤诉求: {'、'.join(self.concerns)}")
        if self.allergies:
            lines.append(f"  需避开: {'、'.join(self.allergies)}")
        if self.climate:
            lines.append(f"  所在气候: {self.climate}")
        return "\n".join(lines)


def aggregate_skin_profile(memories: list[MemoryItem]) -> SkinProfile:
    """从 profile 记忆列表中聚合出结构化的肤质画像。

    将多条 profile 记忆（如"t区油""脸颊干""容易过敏"）合并推演为完整画像。
    """
    profile = SkinProfile()
    texts = [m.text.lower() for m in memories]
    profile.raw_memories = texts[:]

    # ── T区状态 ──
    t_oily = any("t区油" in t or "t区出油" in t or "t区偏油" in t or "t区爱出油" in t for t in texts)
    t_dry = any("t区干" in t or "t区偏干" in t for t in texts)
    if t_oily and t_dry:
        profile.t_zone = "混合（又油又干）"
    elif t_oily:
        profile.t_zone = "偏油"
    elif t_dry:
        profile.t_zone = "偏干"
    elif any("t区" in t for t in texts):
        profile.t_zone = "正常"

    # ── 脸颊状态 ──
    cheek_oily = any("脸颊油" in t or "脸颊偏油" in t or "两颊油" in t or "两颊出油" in t for t in texts)
    cheek_dry = any("脸颊干" in t or "脸颊偏干" in t or "两颊干" in t or "两颊偏干" in t for t in texts)
    if cheek_oily and cheek_dry:
        profile.cheeks = "混合（又油又干）"
    elif cheek_oily:
        profile.cheeks = "偏油"
    elif cheek_dry:
        profile.cheeks = "偏干"
    elif any("脸颊" in t or "两颊" in t for t in texts):
        profile.cheeks = "正常"

    # ── 综合推断肤质类型 ──
    explicit_type = _match_explicit_skin_type(texts)
    if explicit_type:
        profile.skin_type = explicit_type
    elif profile.t_zone == "偏油" and profile.cheeks == "偏干":
        profile.skin_type = "混油"
    elif profile.t_zone == "偏干" and profile.cheeks == "偏油":
        profile.skin_type = "混干"
    elif profile.t_zone == "偏油" and profile.cheeks in (None, "正常"):
        profile.skin_type = "油性"
    elif profile.cheeks == "偏干" and profile.t_zone in (None, "正常"):
        profile.skin_type = "干性"
    elif any("外油内干" in t or "内油外干" in t for t in texts):
        profile.skin_type = "外油内干"
    elif profile.t_zone == "正常" and profile.cheeks == "正常":
        profile.skin_type = "中性"

    # ── 敏感程度 ──
    sensitivity_keywords = ["敏感", "过敏", "容易泛红", "红血丝", "刺痛", "发痒", "屏障受损", "皮肤薄"]
    if any(any(kw in t for kw in sensitivity_keywords) for t in texts):
        profile.sensitivity = "敏感"
    else:
        profile.sensitivity = "一般"

    # ── 痘痘 ──
    acne_keywords = ["痘", "闭口", "粉刺", "爱长痘", "容易长痘", "反复长痘"]
    if any(any(kw in t for kw in acne_keywords) for t in texts):
        profile.acne_prone = True
    elif any("不长痘" in t or "无痘" in t for t in texts):
        profile.acne_prone = False

    # ── 护肤诉求 ──
    concern_map = {
        "美白": ["美白", "变白", "提亮", "暗沉", "色斑", "痘印"],
        "祛痘": ["祛痘", "痘痘", "闭口", "粉刺", "痘印"],
        "抗老": ["抗老", "抗衰", "皱纹", "细纹", "松弛", "紧致"],
        "保湿": ["保湿", "补水", "干燥", "起皮", "脱皮", "干"],
        "修护": ["修护", "修复", "屏障", "泛红", "红血丝", "敏感"],
        "控油": ["控油", "出油", "油光", "油"],
        "舒缓": ["舒缓", "镇静", "退红", "抗炎", "消炎"],
    }
    seen: set[str] = set()
    for concern, keywords in concern_map.items():
        if any(any(kw in t for kw in keywords) for t in texts):
            if concern not in seen:
                profile.concerns.append(concern)
                seen.add(concern)

    # ── 过敏/避开成分 ──
    allergy_keywords = ["过敏", "不耐受", "避开", "不能用", "不能用含"]
    for t in texts:
        if any(kw in t for kw in allergy_keywords):
            for part in t.replace("，", ",").replace("、", ",").split(","):
                part = part.strip()
                if part and ("过敏" not in part) and len(part) > 1:
                    profile.allergies.append(part)

    # ── 气候 ──
    climate_map = {
        "南方": ["南方", "华南", "广州", "深圳", "珠海", "海南", "福州", "厦门"],
        "北方": ["北方", "华北", "北京", "天津", "石家庄", "济南", "青岛", "大连"],
        "干燥": ["干燥", "干", "北方", "西北"],
        "潮湿": ["潮湿", "湿", "回南天", "南方", "华南"],
    }
    for label, keywords in climate_map.items():
        if any(k in t for k in keywords for t in texts):
            profile.climate = label
            break

    return profile


def _match_explicit_skin_type(texts: list[str]) -> str | None:
    """匹配明确声明的肤质类型（如"我是油皮""我是混油"）。"""
    for t in texts:
        if "油皮" in t or "油性" in t:
            if "混" in t or "混合" in t:
                return "混油"
            if "干" not in t:
                return "油皮"
        if "干皮" in t or "干性" in t:
            if "混" in t or "混合" in t:
                return "混干"
            if "油" not in t:
                return "干皮"
        if "中性皮" in t or "中性" in t:
            return "中性"
        if "敏感皮" in t or "敏感性" in t or "敏感肌" in t:
            return "敏感"
        if "痘肌" in t:
            return "油痘肌"
    return None


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
