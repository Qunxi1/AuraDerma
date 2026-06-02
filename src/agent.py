from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from qdrant_client.http import models

from llm import LLMClient
from memory import MemoryBundle, MemoryPolicy, MemoryStore
from prompts import ANSWER_PROMPT, MEMORY_ROUTER_PROMPT, RETRIEVAL_PROMPT, SKILL_ROUTER_PROMPT, SYSTEM_PROMPT
from retrieval import Retriever
from skill_manager import SkillManager
from web_search import WebSearchClient


@dataclass(slots=True)
class AgentContext:
    user_id: str
    question: str
    memory: MemoryBundle


@dataclass(slots=True)
class RetrievalResult:
    product_hits: list
    memory_hits: list
    doc_hits: list
    web_notes: list[str]
    retrieval_plan: str
    used_web: bool
    memory_index_lines: list[str]
    index_recall_lines: list[str]
    open_memory_files: bool
    relevant_memory_ids: list[str]
    skill_plan: str
    skill_names: list[str]
    memory_file_snippets: list[str]
    skill_body: str


class SkincareAgent:
    def __init__(self, llm: LLMClient, retriever: Retriever, web: WebSearchClient, policy: MemoryPolicy) -> None:
        self.llm = llm
        self.retriever = retriever
        self.web = web
        self.policy = policy

    def route(self, ctx: AgentContext, memory_store: MemoryStore, skill_manager: SkillManager) -> RetrievalResult:
        retrieval_plan = self.llm.chat(SYSTEM_PROMPT, f"{RETRIEVAL_PROMPT}\n\n用户问题：{ctx.question}")
        query_vector = self._embed_query(ctx.question)

        product_hits = self.retriever.search(self.retriever.products_collection, query_vector, limit=5)
        memory_hits = self.retriever.search(
            self.retriever.memory_collection,
            query_vector,
            limit=5,
            filters=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=ctx.user_id))]
            ),
        )
        doc_hits = self.retriever.search(self.retriever.docs_collection, query_vector, limit=5)

        memory_index_lines = memory_store.load_index(user_id=ctx.user_id, limit=20)
        index_recall_lines = memory_store.recall_index_lines(user_id=ctx.user_id, query=ctx.question, limit=6)

        router_prompt = dedent(
            f"""
            用户问题:
            {ctx.question}

            记忆索引摘要:
            {chr(10).join(memory_index_lines) if memory_index_lines else '无'}

            索引召回摘要:
            {chr(10).join(index_recall_lines) if index_recall_lines else '无'}

            请判断哪些记忆需要打开原文，返回 JSON。
            """
        ).strip()
        memory_router_raw = self.llm.chat(SYSTEM_PROMPT, f"{MEMORY_ROUTER_PROMPT}\n\n{router_prompt}")
        memory_router = self._parse_json_obj(memory_router_raw)
        relevant_memory_ids = [str(x) for x in memory_router.get("relevant_memory_ids", [])]
        open_memory_files = bool(memory_router.get("open_original_files", False))

        skill_registry_summary = skill_manager.registry_summary()
        skill_prompt = dedent(
            f"""
            用户问题:
            {ctx.question}

            记忆索引摘要:
            {chr(10).join(index_recall_lines) if index_recall_lines else '无'}

            技能注册表摘要:
            {skill_registry_summary}

            内部产品召回:
            {self._format_hits(product_hits)}

            记忆召回:
            {self._format_hits(memory_hits)}

            文档召回:
            {self._format_hits(doc_hits)}

            现在请判断需要调用哪些技能/工具，返回 JSON。
            """
        ).strip()
        skill_router_raw = self.llm.chat(SYSTEM_PROMPT, f"{SKILL_ROUTER_PROMPT}\n\n{skill_prompt}")
        skill_router = self._parse_json_obj(skill_router_raw)
        skill_names = [str(x) for x in skill_router.get("needed_skills", [])]
        skill_plan = json_dumps_pretty(skill_router)
        skill_body = skill_manager.registry_body(skill_names)

        memory_file_snippets = memory_store.read_relevant_memory_texts(
            user_id=ctx.user_id,
            memory_ids=relevant_memory_ids,
            query=ctx.question,
            limit=4,
        ) if open_memory_files else []

        need_web = self._should_use_web(product_hits, memory_hits, doc_hits, index_recall_lines, skill_names)
        web_notes: list[str] = []
        if need_web and self.web.enabled:
            try:
                web_results = self.web.search(ctx.question, top_k=3)
                web_notes = [f"网页搜索参考，仅供参考｜{r.title}｜{r.url}｜{r.snippet}" for r in web_results]
            except Exception:
                web_notes = []

        return RetrievalResult(
            product_hits=product_hits,
            memory_hits=memory_hits,
            doc_hits=doc_hits,
            web_notes=web_notes,
            retrieval_plan=retrieval_plan,
            used_web=bool(web_notes),
            memory_index_lines=memory_index_lines,
            index_recall_lines=index_recall_lines,
            open_memory_files=open_memory_files,
            relevant_memory_ids=relevant_memory_ids,
            skill_plan=skill_plan,
            skill_names=skill_names,
            memory_file_snippets=memory_file_snippets,
            skill_body=skill_body,
        )

    def answer(self, ctx: AgentContext, memory_store: MemoryStore, skill_manager: SkillManager) -> str:
        route = self.route(ctx, memory_store, skill_manager)
        context_block = dedent(
            f"""
            用户问题:
            {ctx.question}

            检索计划:
            {route.retrieval_plan}

            记忆索引:
            {chr(10).join(route.memory_index_lines) if route.memory_index_lines else '无'}

            索引召回:
            {chr(10).join(route.index_recall_lines) if route.index_recall_lines else '无'}

            需要打开的记忆ID:
            {', '.join(route.relevant_memory_ids) if route.relevant_memory_ids else '无'}

            需要调用的技能:
            {', '.join(route.skill_names) if route.skill_names else '无'}

            技能注册表正文:
            {route.skill_body if route.skill_body else '无'}

            技能路由:
            {route.skill_plan}

            打开的记忆原文片段:
            {chr(10).join(route.memory_file_snippets) if route.memory_file_snippets else '无'}

            内部产品召回:
            {self._format_hits(route.product_hits)}

            记忆召回:
            {self._format_hits(route.memory_hits)}

            文档召回:
            {self._format_hits(route.doc_hits)}

            网页搜索参考:
            {chr(10).join(route.web_notes) if route.web_notes else '无'}
            """
        ).strip()
        return self.llm.chat(ANSWER_PROMPT, context_block)

    def auto_memory_extract(self, user_id: str, dialog_text: str) -> list:
        prompt = (
            "请从以下对话中抽取可长期保存的护肤记忆，返回 JSON 数组。"
            "每项包含 text, scope(profile|long_term), confidence, tags。"
            "只返回 JSON，不要解释。\n\n"
            f"对话内容：\n{dialog_text}"
        )
        raw = self.llm.chat(SYSTEM_PROMPT, prompt, temperature=0.0)
        items = self._parse_json_list(raw)
        memories = []
        for item in items:
            scope = item.get("scope", "long_term")
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            mem = self.policy.classify(text, user_id=user_id)
            if scope == "profile":
                mem.scope = mem.scope.PROFILE
                mem.ttl_days = None
                if "profile" not in mem.tags:
                    mem.tags.append("profile")
            else:
                mem.scope = mem.scope.LONG_TERM
                if "long_term" not in mem.tags:
                    mem.tags.append("long_term")
            mem.confidence = float(item.get("confidence", mem.confidence))
            mem.tags = list(dict.fromkeys([*mem.tags, *item.get("tags", [])]))
            memories.append(mem)
        return memories

    def finalize_turn(self, user_id: str, dialog_text: str, memory_store: MemoryStore, memory_bundle: MemoryBundle) -> None:
        extracted = self.auto_memory_extract(user_id=user_id, dialog_text=dialog_text)
        for item in extracted:
            memory_store.append(item)
            self._apply_memory_to_bundle(memory_bundle, item)

    def _should_use_web(self, product_hits, memory_hits, doc_hits, index_recall_lines, skill_names) -> bool:
        if product_hits:
            return False
        if doc_hits:
            return False
        if memory_hits:
            return False
        if index_recall_lines:
            return False
        return "web_search" in skill_names or len(skill_names) == 0

    def _embed_query(self, question: str) -> list[float]:
        return self.llm.embed([question])[0]

    @staticmethod
    def _format_hits(hits) -> str:
        if not hits:
            return "无"
        lines = []
        for hit in hits:
            payload = hit.payload
            title = payload.get("name") or payload.get("title") or payload.get("summary") or payload.get("text", "")[:80]
            lines.append(f"- score={hit.score:.3f} id={hit.id} {title}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_list(raw: str) -> list[dict]:
        import json

        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _parse_json_obj(raw: str) -> dict:
        import json

        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _apply_memory_to_bundle(bundle: MemoryBundle, item) -> None:
        if item.scope.value == "profile":
            bundle.profile.append(item)
        elif item.scope.value == "short_term":
            bundle.short_term.append(item)
        elif item.scope.value == "long_term":
            bundle.long_term.append(item)
        else:
            bundle.case_notes.append(item)


def json_dumps_pretty(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
