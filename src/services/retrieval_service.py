from __future__ import annotations

from qdrant_client.http import models as qdrant_models

from schema import AgentContext, IntentResult, RegimenPlan, WorkflowPlan, json_dumps_pretty
from core import JsonParser, get_logger
from prompts import MEMORY_ROUTER_PROMPT, RETRIEVAL_PROMPT, SKILL_ROUTER_PROMPT, SYSTEM_PROMPT

log = get_logger("auraderma.retrieval")


class RetrievalService:
    """检索服务。

    封装产品搜索、记忆检索、文档检索、技能路由等多种检索策略。
    """

    def __init__(self, llm, retriever, embedder=None):
        self._llm = llm
        self._retriever = retriever
        self._embedder = embedder

    # ------------------------------------------------------------------
    # 嵌入向量
    # ------------------------------------------------------------------

    def embed_query(self, question: str) -> list[float]:
        if self._embedder is not None:
            return self._embedder.embed([question])[0]
        return self._llm.embed([question])[0]

    # ------------------------------------------------------------------
    # 通用检索
    # ------------------------------------------------------------------

    def search_memory(self, embedding: list[float], user_id: str, limit: int = 5):
        return self._retriever.search(
            self._retriever.memory_collection,
            embedding,
            limit=limit,
            filters=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="user_id",
                        match=qdrant_models.MatchValue(value=user_id),
                    )
                ]
            ),
        )

    def search_docs(self, embedding: list[float], limit: int = 5):
        return self._retriever.search(self._retriever.docs_collection, embedding, limit=limit)

    def search_products(self, embedding: list[float], limit: int = 8):
        return self._retriever.search(self._retriever.products_collection, embedding, limit=limit)

    # ------------------------------------------------------------------
    # 记忆路由
    # ------------------------------------------------------------------

    def route_memory(
        self,
        question: str,
        memory_index_lines: list[str],
        index_recall_lines: list[str],
    ):
        """判断哪些记忆需要打开原文。"""
        log.info("执行记忆路由")

        router_prompt = (
            f"用户问题:\n{question}\n\n"
            f"记忆索引摘要:\n{chr(10).join(memory_index_lines) if memory_index_lines else '无'}\n\n"
            f"索引召回摘要:\n{chr(10).join(index_recall_lines) if index_recall_lines else '无'}\n\n"
            "请判断哪些记忆需要打开原文，返回 JSON。"
        )
        raw = self._llm.chat(SYSTEM_PROMPT, f"{MEMORY_ROUTER_PROMPT}\n\n{router_prompt}")
        obj = JsonParser.parse_obj(raw, context="memory_router")

        relevant_ids = [str(x) for x in obj.get("relevant_memory_ids", [])]
        open_files = bool(obj.get("open_original_files", False))
        log.info("记忆路由结果: %d 条相关, open=%s", len(relevant_ids), open_files)
        return relevant_ids, open_files

    # ------------------------------------------------------------------
    # 产品检索策略
    # ------------------------------------------------------------------

    def search_single_product(self, question: str, embedding: list[float], categories: list[str]):
        """单品检索模式。"""
        log.info("单品检索: categories=%s", categories)
        retrieval_plan = self._llm.chat(
            SYSTEM_PROMPT,
            f"{RETRIEVAL_PROMPT}\n\n用户问题：{question}",
        )
        if categories:
            enhanced_query = f"{question} {' '.join(categories)}"
            enhanced_emb = self.embed_query(enhanced_query)
            hits = self.search_products(enhanced_emb, limit=8)
        else:
            hits = self.search_products(embedding, limit=5)
        return retrieval_plan, hits

    def search_multi_category(self, question: str, categories: list[str]):
        """多品类检索模式。"""
        log.info("多品类检索: categories=%s", categories)
        multi_hits: dict[str, list] = {}
        for cat in categories:
            cat_query = f"{question} {cat} 推荐"
            cat_emb = self.embed_query(cat_query)
            multi_hits[cat] = self._retriever.search(
                self._retriever.products_collection, cat_emb, limit=4,
            )

        # 去重合并
        all_ids: set[str] = set()
        merged: list = []
        for hits in multi_hits.values():
            for h in hits:
                if h.id not in all_ids:
                    all_ids.add(h.id)
                    merged.append(h)

        retrieval_plan = f"[多品类模式] 用户指定品类：{'、'.join(categories)}"
        return retrieval_plan, merged, multi_hits

    def search_regimen_products(self, question: str, regimen_plan: RegimenPlan):
        """护肤体系检索模式：按步骤逐一检索。"""
        log.info("护肤体系检索: %d 个步骤", len(regimen_plan.steps))

        seen_queries: dict[str, list] = {}
        for step in regimen_plan.steps:
            sq = step.search_query
            if sq not in seen_queries:
                sq_emb = self.embed_query(sq)
                seen_queries[sq] = self._retriever.search(
                    self._retriever.products_collection, sq_emb, limit=3,
                )
            step.product_hits = seen_queries[sq]

        all_ids: set[str] = set()
        merged: list = []
        for hits in seen_queries.values():
            for h in hits:
                if h.id not in all_ids:
                    all_ids.add(h.id)
                    merged.append(h)

        retrieval_plan = (
            f"[护肤体系模式] 目标={regimen_plan.goal}，"
            f"规划了 {len(regimen_plan.steps)} 个步骤，"
            f"覆盖品类：{'、'.join(dict.fromkeys(s.category for s in regimen_plan.steps))}"
        )
        return retrieval_plan, merged, seen_queries

    # ------------------------------------------------------------------
    # 技能路由
    # ------------------------------------------------------------------

    def route_skills(self, question: str, intent: IntentResult, memory_lines: list[str],
                     recall_lines: list[str], product_hits: list, memory_hits: list,
                     doc_hits: list, skill_manager) -> tuple[list[str], str, str]:
        """路由需要调用的外部技能。"""
        log.info("路由外部技能")

        skill_registry_summary = skill_manager.registry_summary()
        skill_prompt = (
            f"用户问题:\n{question}\n\n"
            f"意图: {intent.intent}, 目标: {intent.goal}\n\n"
            f"记忆索引摘要:\n{chr(10).join(recall_lines) if recall_lines else '无'}\n\n"
            f"技能注册表摘要:\n{skill_registry_summary}\n\n"
            f"内部产品召回:\n{self._format_hits(product_hits)}\n\n"
            f"记忆召回:\n{self._format_hits(memory_hits)}\n\n"
            f"文档召回:\n{self._format_hits(doc_hits)}\n\n"
            "现在请判断需要调用哪些技能/工具，返回 JSON。"
        )
        raw = self._llm.chat(SYSTEM_PROMPT, f"{SKILL_ROUTER_PROMPT}\n\n{skill_prompt}")
        obj = JsonParser.parse_obj(raw, context="skill_router")

        skill_names = [str(x) for x in obj.get("needed_skills", [])]
        skill_plan = json_dumps_pretty(obj)
        skill_body = skill_manager.registry_body(skill_names)
        log.info("技能路由结果: %s", skill_names)
        return skill_names, skill_plan, skill_body

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def should_use_web(product_hits, memory_hits, doc_hits,
                       index_recall_lines, skill_names) -> bool:
        if product_hits or doc_hits or memory_hits or index_recall_lines:
            return False
        return "web_search" in skill_names or len(skill_names) == 0

    @staticmethod
    def _format_hits(hits) -> str:
        if not hits:
            return "无"
        lines = []
        for hit in hits:
            payload = hit.payload
            title = (
                payload.get("name")
                or payload.get("title")
                or payload.get("summary")
                or payload.get("text", "")[:80]
            )
            lines.append(f"- score={hit.score:.3f} id={hit.id} {title}")
        return "\n".join(lines)
