from __future__ import annotations

from dataclasses import asdict, dataclass, field
from textwrap import dedent

from qdrant_client.http import models

from llm import LLMClient
from memory import MemoryBundle, MemoryPolicy, MemoryStore
from prompts import (
    ANSWER_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    MEMORY_ROUTER_PROMPT,
    REGIMEN_PLANNER_PROMPT,
    RETRIEVAL_PROMPT,
    SKILL_ROUTER_PROMPT,
    SYSTEM_PROMPT,
    WEATHER_EXTRACT_PROMPT,
    WORKFLOW_PLANNER_PROMPT,
)
from reporter import NullReporter, ProgressReporter

# ---------------------------------------------------------------------------
# Answer mode instruction templates (used to fill ANSWER_PROMPT placeholders)
# ---------------------------------------------------------------------------

MODE_INSTRUCTIONS: dict[str, str] = {
    "general_chat": (
        "Structure:\n"
        "1. Respond directly to the user's question in a natural, helpful way.\n"
        "2. Do NOT mention skincare, beauty products, or any skincare-related advice.\n"
        "3. Do NOT recommend any products.\n"
        "4. If the user asks about general knowledge, provide a clear and concise answer.\n"
        "5. If the user is chit-chatting, respond in a friendly and engaging manner.\n"
        "6. Do NOT ask follow-up questions about skin or beauty.\n\n"
        "IMPORTANT: This is a general conversation mode. The user is NOT asking about skincare. "
        "Keep the response focused on their actual question only."
    ),
    "skincare_analysis": (
        "Structure:\n"
        "1. Analyze the user's skin concern based on general dermatological knowledge.\n"
        "2. Explain possible causes (e.g. lifestyle, genetics, environment).\n"
        "3. Provide medical disclaimer: this is not a diagnosis, consult a dermatologist for serious concerns.\n"
        "4. If the user's question naturally leads to skincare advice, you MAY briefly mention "
        "that certain types of products could help, but do NOT give specific product names.\n"
        "5. One or two follow-up questions about their skin condition if appropriate.\n\n"
        "Style: Educational and informative. Do NOT hard-sell any products."
    ),
    "product_search": (
        "Structure:\n"
        "1. Short assessment of the skin situation.\n"
        "2. Suggested routine or treatment direction.\n"
        "3. Product recommendations from the knowledge base.\n"
        "4. If needed, web-sourced suggestions clearly marked as \"网页搜索参考，仅供参考\".\n"
        "5. Cautions / contraindications.\n"
        "6. One or two follow-up questions if needed.\n\n"
        "IMPORTANT: Only recommend products that match the user's stated concerns."
    ),
    "regimen_planning": (
        "Structure:\n"
        "1. Short assessment of the skin situation.\n"
        "2. Suggested routine or treatment direction.\n"
        "3. Product recommendations from the knowledge base.\n"
        "4. If needed, web-sourced suggestions clearly marked as \"网页搜索参考，仅供参考\".\n"
        "5. Cautions / contraindications.\n"
        "6. One or two follow-up questions if needed.\n\n"
        "IMPORTANT: Present the regimen as a complete routine with morning/evening/periodic steps."
    ),
}

DEFAULT_INSTRUCTIONS = (
    "Structure:\n"
    "1. Respond directly and helpfully to the user.\n"
    "2. Be concise and clear.\n"
    "3. Do not fabricate information.\n"
    "4. If you don't know something, say so.\n"
)

from retrieval import Retriever
from skill_manager import SkillManager
from web_search import WebSearchClient


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentContext:
    user_id: str
    question: str
    memory: MemoryBundle


@dataclass(slots=True)
class IntentResult:
    intent: str  # "single" | "multi" | "regimen" | "general"
    goal: str
    has_explicit_category: bool
    explicit_categories: list[str] = field(default_factory=list)
    reasoning: str = ""
    is_skincare_related: bool = True


@dataclass(slots=True)
class WorkflowPlan:
    """Which processes to execute, decided by the LLM workflow planner."""
    processes: list[str] = field(default_factory=list)
    rationale: str = ""
    needs_product_search: bool = False
    needs_skincare_advice: bool = False


@dataclass(slots=True)
class RegimenStep:
    category: str
    purpose: str
    search_query: str
    time_of_day: str  # "morning" | "evening" | "periodic"
    product_hits: list = field(default_factory=list)


@dataclass(slots=True)
class RegimenPlan:
    goal: str
    goal_explanation: str
    steps: list[RegimenStep] = field(default_factory=list)
    must_have_categories: list[str] = field(default_factory=list)
    avoid_ingredients: list[str] = field(default_factory=list)
    notes: str = ""
    category_priority: list[str] = field(default_factory=list)

    # Grouped access
    @property
    def morning_steps(self) -> list[RegimenStep]:
        return [s for s in self.steps if s.time_of_day == "morning"]

    @property
    def evening_steps(self) -> list[RegimenStep]:
        return [s for s in self.steps if s.time_of_day == "evening"]

    @property
    def periodic_steps(self) -> list[RegimenStep]:
        return [s for s in self.steps if s.time_of_day == "periodic"]


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
    weather_info: str = ""  # formatted weather data or empty
    # --- intent-aware fields ---
    intent: str = "single"
    regimen_goal: str = ""
    regimen_plan: RegimenPlan | None = None
    multi_category_hits: dict[str, list] = field(default_factory=dict)
    # --- workflow-plan fields ---
    workflow_plan: WorkflowPlan | None = None
    is_general_chat: bool = False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SkincareAgent:
    def __init__(self, llm: LLMClient, retriever: Retriever, web: WebSearchClient, policy: MemoryPolicy, _embedder: object | None = None, reporter: ProgressReporter | None = None) -> None:
        self.llm = llm
        self.retriever = retriever
        self.web = web
        self.policy = policy
        self._embedder = _embedder  # 可选的外部 embedder（如 LocalEmbedder），为 None 时回退到 self.llm.embed()
        self.reporter = reporter or NullReporter()

    # ------------------------------------------------------------------
    # Intent classification + regimen planning
    # ------------------------------------------------------------------

    def classify_intent(self, question: str) -> IntentResult:
        self.reporter.intent()
        raw = self.llm.chat(SYSTEM_PROMPT, f"{INTENT_CLASSIFIER_PROMPT}\n\n用户问题：{question}")
        obj = self._parse_json_obj(raw)
        return IntentResult(
            intent=obj.get("intent", "single"),
            goal=obj.get("goal", "护肤咨询"),
            has_explicit_category=bool(obj.get("has_explicit_category", False)),
            explicit_categories=[str(c) for c in obj.get("explicit_categories", [])],
            reasoning=str(obj.get("reasoning", "")),
            is_skincare_related=bool(obj.get("is_skincare_related", True)),
        )

    def plan_regimen(self, question: str, goal: str) -> RegimenPlan:
        self.reporter.regimen()
        prompt = dedent(
            f"""
            {REGIMEN_PLANNER_PROMPT}

            用户问题：
            {question}

            识别出的护肤目标：{goal}

            请规划护肤体系，返回 JSON。
            """
        ).strip()
        raw = self.llm.chat(SYSTEM_PROMPT, prompt)
        obj = self._parse_json_obj(raw)

        # Flatten all step groups into one list, keeping time_of_day
        steps: list[RegimenStep] = []
        for time_slot, key in [("morning", "morning_steps"), ("evening", "evening_steps"), ("periodic", "periodic_steps")]:
            for step_data in obj.get(key, []):
                steps.append(RegimenStep(
                    category=str(step_data.get("category", "")),
                    purpose=str(step_data.get("purpose", "")),
                    search_query=str(step_data.get("search_query", "")),
                    time_of_day=time_slot,
                ))

        return RegimenPlan(
            goal=goal,
            goal_explanation=str(obj.get("goal_explanation", "")),
            steps=steps,
            must_have_categories=[str(c) for c in obj.get("must_have_categories", [])],
            avoid_ingredients=[str(i) for i in obj.get("avoid_ingredients", [])],
            notes=str(obj.get("notes", "")),
            category_priority=[str(c) for c in obj.get("category_priority", [])],
        )

    def plan_workflows(self, question: str, intent: IntentResult) -> WorkflowPlan:
        """Let the LLM decide which processes to execute based on intent + question."""
        self.reporter.workflow()
        prompt = dedent(
            f"""
            用户问题:
            {question}

            意图分类结果:
            - intent: {intent.intent}
            - goal: {intent.goal}
            - skincare_related: {intent.is_skincare_related}
            - reasoning: {intent.reasoning}

            请规划要执行的流程，返回 JSON。
            """
        ).strip()
        raw = self.llm.chat(SYSTEM_PROMPT, f"{WORKFLOW_PLANNER_PROMPT}\n\n{prompt}")
        obj = self._parse_json_obj(raw)
        processes = [str(p) for p in obj.get("processes", [])]

        # Safety net: general intent must have general_chat (but may also have non-skincare skills)
        if intent.intent == "general" and "general_chat" not in processes:
            processes.insert(0, "general_chat")

        # Safety net: general intent should never have skincare processes
        skincare_processes = {"product_search", "skincare_analysis", "regimen_planning", "memory_lookup"}
        if intent.intent == "general":
            processes = [p for p in processes if p not in skincare_processes]
            if "general_chat" not in processes:
                processes.insert(0, "general_chat")

        return WorkflowPlan(
            processes=processes,
            rationale=str(obj.get("rationale", "")),
            needs_product_search=bool(obj.get("needs_product_search", False)),
            needs_skincare_advice=bool(obj.get("needs_skincare_advice", "skincare_analysis" in processes)),
        )

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------

    def route(self, ctx: AgentContext, memory_store: MemoryStore, skill_manager: SkillManager) -> RetrievalResult:
        # ① 意图分类
        intent = self.classify_intent(ctx.question)

        # ①½ 工作流规划：让 LLM 决定走哪些流程
        workflow = self.plan_workflows(ctx.question, intent)

        # ── 如果是纯聊天模式（非护肤相关），跳过护肤流程，但仍可执行辅助技能 ──
        if "general_chat" in workflow.processes:
            # 仍可执行非护肤辅助技能
            weather_info = ""
            if "weather_check" in workflow.processes:
                weather_info = self._fetch_weather(ctx, memory_store, skill_manager)

            need_web = "web_search" in workflow.processes
            web_notes: list[str] = []
            if need_web and self.web.enabled:
                self.reporter.web_search(ctx.question)
                try:
                    web_results = self.web.search(ctx.question, top_k=3)
                    web_notes = [f"网页搜索参考，仅供参考｜{r.title}｜{r.url}｜{r.snippet}" for r in web_results]
                except Exception:
                    web_notes = []

            return RetrievalResult(
                product_hits=[],
                memory_hits=[],
                doc_hits=[],
                web_notes=web_notes,
                retrieval_plan="[纯聊天模式] 不涉及护肤相关处理",
                used_web=bool(web_notes),
                memory_index_lines=[],
                index_recall_lines=[],
                open_memory_files=False,
                relevant_memory_ids=[],
                skill_plan="",
                skill_names=["web_search" if need_web else ""],
                memory_file_snippets=[],
                skill_body="",
                weather_info=weather_info,
                intent=intent.intent,
                regimen_goal=intent.goal,
                regimen_plan=None,
                multi_category_hits={},
                workflow_plan=workflow,
                is_general_chat=True,
            )

        # ── 护肤相关流程 ──
        question_embedding = self._embed_query(ctx.question)

        # ② 通用检索（memory / docs 共用）
        memory_hits = self.retriever.search(
            self.retriever.memory_collection,
            question_embedding,
            limit=5,
            filters=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=ctx.user_id))]
            ),
        )
        doc_hits = self.retriever.search(self.retriever.docs_collection, question_embedding, limit=5)

        memory_index_lines = memory_store.load_index(user_id=ctx.user_id, limit=20)
        index_recall_lines = memory_store.recall_index_lines(user_id=ctx.user_id, query=ctx.question, limit=6)

        # ③ Memory router (仅当工作流包含 memory_lookup)
        relevant_memory_ids: list[str] = []
        open_memory_files = False
        memory_file_snippets: list[str] = []

        if "memory_lookup" in workflow.processes:
            self.reporter.memory()
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

            memory_file_snippets = memory_store.read_relevant_memory_texts(
                user_id=ctx.user_id,
                memory_ids=relevant_memory_ids,
                query=ctx.question,
                limit=4,
            ) if open_memory_files else []

        # ④ 产品检索：按工作流规划分路
        retrieval_plan: str = ""
        product_hits: list = []
        regimen_plan: RegimenPlan | None = None
        multi_category_hits: dict[str, list] = {}

        if "product_search" in workflow.processes or "regimen_planning" in workflow.processes:
            if intent.intent == "regimen" or "regimen_planning" in workflow.processes:
                # ── 护肤体系模式：LLM 规划品类 → 逐品类 RAG ──
                regimen_plan = self.plan_regimen(ctx.question, intent.goal)

                seen_queries: dict[str, list] = {}
                for step in regimen_plan.steps:
                    sq = step.search_query
                    if sq not in seen_queries:
                        sq_emb = self._embed_query(sq)
                        seen_queries[sq] = self.retriever.search(
                            self.retriever.products_collection, sq_emb, limit=3
                        )
                    step.product_hits = seen_queries[sq]

                all_hit_ids: set[str] = set()
                merged_hits: list = []
                for hits in seen_queries.values():
                    for h in hits:
                        if h.id not in all_hit_ids:
                            all_hit_ids.add(h.id)
                            merged_hits.append(h)
                product_hits = merged_hits

                retrieval_plan = (
                    f"[护肤体系模式] 目标={intent.goal}，"
                    f"规划了 {len(regimen_plan.steps)} 个步骤，"
                    f"覆盖品类：{'、'.join(dict.fromkeys(s.category for s in regimen_plan.steps))}"
                )

            elif intent.intent == "multi" and len(intent.explicit_categories) >= 2:
                # ── 多品类模式 ──
                self.reporter.product_search("、".join(intent.explicit_categories))
                for cat in intent.explicit_categories:
                    cat_query = f"{ctx.question} {cat} 推荐"
                    cat_emb = self._embed_query(cat_query)
                    cat_hits = self.retriever.search(
                        self.retriever.products_collection, cat_emb, limit=4
                    )
                    multi_category_hits[cat] = cat_hits

                all_hit_ids = set()
                merged_hits = []
                for hits in multi_category_hits.values():
                    for h in hits:
                        if h.id not in all_hit_ids:
                            all_hit_ids.add(h.id)
                            merged_hits.append(h)
                product_hits = merged_hits

                retrieval_plan = (
                    f"[多品类模式] 用户指定品类：{'、'.join(intent.explicit_categories)}"
                )

            else:
                # ── 单品类模式 ──
                self.reporter.product_search()
                retrieval_plan = self.llm.chat(SYSTEM_PROMPT, f"{RETRIEVAL_PROMPT}\n\n用户问题：{ctx.question}")
                if intent.explicit_categories:
                    enhanced_query = f"{ctx.question} {' '.join(intent.explicit_categories)}"
                    enhanced_emb = self._embed_query(enhanced_query)
                    product_hits = self.retriever.search(
                        self.retriever.products_collection, enhanced_emb, limit=8
                    )
                else:
                    product_hits = self.retriever.search(
                        self.retriever.products_collection, question_embedding, limit=5
                    )

        # ⑤ Skill router (仅当工作流需要外部技能时)
        skill_names: list[str] = []
        skill_plan = ""
        skill_body = ""

        if "web_search" in workflow.processes or "weather_check" in workflow.processes or "file_read" in workflow.processes:
            skill_registry_summary = skill_manager.registry_summary()
            skill_prompt = dedent(
                f"""
                用户问题:
                {ctx.question}

                意图: {intent.intent}, 目标: {intent.goal}

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

        # ⑥ Weather check
        weather_info = ""
        if "weather_check" in workflow.processes or "weather_check" in skill_names:
            weather_info = self._fetch_weather(ctx, memory_store, skill_manager)

        # ⑦ Web search
        need_web = (
            "web_search" in workflow.processes
            and self._should_use_web(product_hits, memory_hits, doc_hits, index_recall_lines, skill_names)
        )
        web_notes: list[str] = []
        if need_web and self.web.enabled:
            self.reporter.web_search(ctx.question)
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
            weather_info=weather_info,
            intent=intent.intent,
            regimen_goal=intent.goal,
            regimen_plan=regimen_plan,
            multi_category_hits=multi_category_hits,
            workflow_plan=workflow,
            is_general_chat=False,
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def answer(self, ctx: AgentContext, memory_store: MemoryStore, skill_manager: SkillManager) -> str:
        route = self.route(ctx, memory_store, skill_manager)

        # 纯聊天模式：不涉及任何护肤内容
        if route.is_general_chat:
            return self._answer_general(ctx, route)

        # 护肤相关模式：根据执行的工作流选择回答模板
        self.reporter.answer()
        if route.intent == "regimen" and route.regimen_plan:
            return self._answer_regimen(ctx, route)
        elif route.intent == "multi" and route.multi_category_hits:
            return self._answer_multi(ctx, route)
        else:
            return self._answer_single(ctx, route)

    def _answer_single(self, ctx: AgentContext, route: RetrievalResult) -> str:
        # Determine the best mode instructions
        mode = "product_search"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        # If the workflow plan suggests skincare analysis without product focus, adjust
        if route.workflow_plan and not route.workflow_plan.needs_product_search:
            mode = "skincare_analysis"
            mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        answer_prompt_filled = ANSWER_PROMPT.format(mode=mode, mode_instructions=mode_instructions)

        context_block = dedent(
            f"""
            用户问题:
            {ctx.question}

            意图模式: {route.intent} | 护肤目标: {route.regimen_goal}

            检索计划:
            {route.retrieval_plan}

            工作流规划:
            {json_dumps_pretty(asdict(route.workflow_plan)) if route.workflow_plan else '无'}

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

            当地气候数据:
            {route.weather_info if route.weather_info else '无'}
            """
        ).strip()
        return self.llm.chat(answer_prompt_filled, context_block)

    def _answer_multi(self, ctx: AgentContext, route: RetrievalResult) -> str:
        """为 multi 模式生成回答：按品类分组展示推荐"""
        mode = "product_search"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)
        answer_prompt_filled = ANSWER_PROMPT.format(mode=mode, mode_instructions=mode_instructions)

        category_blocks: list[str] = []
        for cat, hits in route.multi_category_hits.items():
            block = f"\n【品类：{cat}】"
            if hits:
                block += f"\n  匹配到 {len(hits)} 款产品："
                for h in hits:
                    p = h.payload
                    name = p.get("name", "?")
                    brand = p.get("brand", "?")
                    price = p.get("price_cny", "")
                    price_str = f" (¥{price})" if price else ""
                    concerns = "、".join(p.get("concerns", [])[:4])
                    block += f"\n  · [{brand}] {name}{price_str} — {concerns}  [score={h.score:.2f}]"
            else:
                block += "\n  (当前知识库暂无该品类产品)"
            category_blocks.append(block)

        context_block = dedent(
            f"""
            用户问题:
            {ctx.question}

            意图模式: multi
            用户指定品类: {'、'.join(route.multi_category_hits.keys())}

            ── 按品类推荐 ──
            {''.join(category_blocks)}

            记忆索引:
            {chr(10).join(route.memory_index_lines) if route.memory_index_lines else '无'}

            需要打开的记忆ID:
            {', '.join(route.relevant_memory_ids) if route.relevant_memory_ids else '无'}

            打开的记忆原文片段:
            {chr(10).join(route.memory_file_snippets) if route.memory_file_snippets else '无'}

            网页搜索参考:
            {chr(10).join(route.web_notes) if route.web_notes else '无'}

            当地气候数据:
            {route.weather_info if route.weather_info else '无'}

            ── 回答要求 ──
            请按品类分类回答，每个品类下列出对应的产品推荐。
            说明每个产品的品牌、价格，以及为什么适合用户。
            如果某品类内部无产品，明确说"当前知识库暂无该品类产品"，不要编造。
            最后可以给出搭配建议和使用先后顺序（如有需要）。
            """
        ).strip()

        return self.llm.chat(answer_prompt_filled, context_block)

    def _answer_general(self, ctx: AgentContext, route: RetrievalResult) -> str:
        """纯聊天模式：不涉及任何护肤内容，直接回答问题。
        但如果通过辅助技能（天气、网页搜索）获取了数据，一并提供给 LLM。
        """
        self.reporter.answer()
        mode = "general_chat"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)
        answer_prompt_filled = ANSWER_PROMPT.format(mode=mode, mode_instructions=mode_instructions)

        weather_section = f"当地气候数据:\n{route.weather_info}" if route.weather_info else ""
        web_section = f"网页搜索参考:\n{chr(10).join(route.web_notes)}" if route.web_notes else ""

        context_block = dedent(
            f"""
            用户问题:
            {ctx.question}

            这是一次纯聊天对话。不要涉及任何护肤、美容、产品推荐相关的内容。
            请直接用中文回答用户的问题，保持友好和帮助性。

            {weather_section}
            {web_section}
            """
        ).strip()
        return self.llm.chat(answer_prompt_filled, context_block)

    def _answer_regimen(self, ctx: AgentContext, route: RetrievalResult) -> str:
        plan = route.regimen_plan
        assert plan is not None

        mode = "regimen_planning"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)
        answer_prompt_filled = ANSWER_PROMPT.format(mode=mode, mode_instructions=mode_instructions)

        # 构建按时间分组的产品推荐块
        regimen_blocks: list[str] = []

        def _format_time_group(label: str, steps: list[RegimenStep]) -> str:
            if not steps:
                return ""
            lines = [f"\n【{label}】"]
            for step in steps:
                lines.append(f"\n  ▶ {step.category} — {step.purpose}")
                lines.append(f"     检索词: {step.search_query}")
                if step.product_hits:
                    lines.append(f"     内部召回 ({len(step.product_hits)} 款):")
                    for h in step.product_hits:
                        p = h.payload
                        name = p.get("name", "?")
                        brand = p.get("brand", "?")
                        price = p.get("price_cny", "")
                        price_str = f" ¥{price}" if price else ""
                        concerns = "、".join(p.get("concerns", [])[:4])
                        lines.append(f"       score={h.score:.2f} | [{brand}] {name}{price_str} | {concerns}")
                else:
                    lines.append("     (内部暂无匹配产品，建议网页搜索补充)")
            return "\n".join(lines)

        regimen_blocks.append(_format_time_group("☀️ 日间护理", plan.morning_steps))
        regimen_blocks.append(_format_time_group("🌙 夜间护理", plan.evening_steps))
        regimen_blocks.append(_format_time_group("📅 周期护理", plan.periodic_steps))

        must_have_str = "、".join(plan.must_have_categories) if plan.must_have_categories else "无"
        avoid_str = "、".join(plan.avoid_ingredients) if plan.avoid_ingredients else "无"

        context_block = dedent(
            f"""
            用户问题:
            {ctx.question}

            意图模式: regimen | 护肤目标: {plan.goal}
            目标说明: {plan.goal_explanation}

            ── 护肤体系规划 ──
            {''.join(regimen_blocks)}

            ── 体系约束 ──
            必须覆盖的品类: {must_have_str}
            建议避免的成分: {avoid_str}
            品类优先级: {'、'.join(plan.category_priority) if plan.category_priority else '无'}
            备注: {plan.notes if plan.notes else '无'}

            记忆索引:
            {chr(10).join(route.memory_index_lines) if route.memory_index_lines else '无'}

            需要打开的记忆ID:
            {', '.join(route.relevant_memory_ids) if route.relevant_memory_ids else '无'}

            打开的记忆原文片段:
            {chr(10).join(route.memory_file_snippets) if route.memory_file_snippets else '无'}

            网页搜索参考:
            {chr(10).join(route.web_notes) if route.web_notes else '无'}

            当地气候数据:
            {route.weather_info if route.weather_info else '无'}

            ── 回答要求 ──
            请按照日间→夜间→周期护理的时间线组织回答。
            每个步骤先说明目的，再列出内部召回的产品推荐，最后如需补充可提及网页搜索。
            如果某品类内部无产品，明确指出"当前知识库暂无该品类产品"，不要编造。
            最后给出完整的使用流程总结和注意事项。
            """
        ).strip()

        return self.llm.chat(answer_prompt_filled, context_block)

    # ------------------------------------------------------------------
    # Memory / helpers
    # ------------------------------------------------------------------

    def auto_memory_extract(self, user_id: str, dialog_text: str) -> list:
        prompt = (
            "请从以下对话中抽取可长期保存的护肤记忆，返回 JSON 数组。"
            "每项包含 text, scope(profile|long_term), confidence(0~1之间的数字，如0.8), tags。"
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
            mem.confidence = self._parse_confidence(item.get("confidence", mem.confidence))
            mem.tags = list(dict.fromkeys([*mem.tags, *item.get("tags", [])]))
            memories.append(mem)
        return memories

    def finalize_turn(self, user_id: str, dialog_text: str, memory_store: MemoryStore, memory_bundle: MemoryBundle) -> None:
        extracted = self.auto_memory_extract(user_id=user_id, dialog_text=dialog_text)
        for item in extracted:
            memory_store.append(item)
            self._apply_memory_to_bundle(memory_bundle, item)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _fetch_weather(self, ctx: AgentContext, memory_store: MemoryStore, skill_manager: SkillManager) -> str:
        """Extract city from question/memory and fetch weather data."""
        self.reporter.thinking("城市识别")
        # 构建 memory profile 摘要帮助 LLM 判断城市
        profile_lines = ctx.memory.summary_lines(max_items_per_scope=20)

        extract_prompt = dedent(
            f"""
            用户问题:
            {ctx.question}

            用户画像记忆:
            {chr(10).join(profile_lines) if profile_lines else '无'}

            请判断用户所在城市，返回 JSON。
            """
        ).strip()
        raw = self.llm.chat(
            SYSTEM_PROMPT,
            f"{WEATHER_EXTRACT_PROMPT}\n\n{extract_prompt}",
            temperature=0.0,
        )
        obj = self._parse_json_obj(raw)
        city = obj.get("city")
        if not city:
            return ""

        self.reporter.weather(city)
        try:
            weather = skill_manager.weather.fetch(city)
            return (
                f"用户所在城市：{weather.get('city', city)}"
                f"{' (' + weather['region'] + ')' if weather.get('region') else ''}\n"
                f"当前气温：{weather['temperature']}°C"
                f" (体感 {weather['feels_like']}°C)\n"
                f"当前湿度：{weather['humidity']}%\n"
                f"天气状况：{weather['condition']}"
            )
        except Exception as e:
            return f"天气查询失败: {e}"

    def _embed_query(self, question: str) -> list[float]:
        if self._embedder is not None:
            return self._embedder.embed([question])[0]  # type: ignore[union-attr]
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
    def _parse_confidence(value: object, default: float = 0.5) -> float:
        """解析 confidence 值，兼容 LLM 返回 'high'/'medium'/'low' 等字符串。"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            mapping = {"high": 0.9, "medium": 0.6, "low": 0.3, "very high": 1.0, "very low": 0.1}
            return mapping.get(lower, default)
        return default

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
