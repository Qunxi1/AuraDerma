from __future__ import annotations

import json
from dataclasses import dataclass, field
from textwrap import dedent

from qdrant_client.http import models as qdrant_models

from core import JsonParser, get_logger
from llm import LLMClient
from memory import MemoryBundle, MemoryPolicy, MemoryStore
from prompts import (
    MEMORY_ROUTER_PROMPT,
    RETRIEVAL_PROMPT,
    SKILL_ROUTER_PROMPT,
    SYSTEM_PROMPT,
    WEATHER_EXTRACT_PROMPT,
    WORKFLOW_PLANNER_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    REGIMEN_PLANNER_PROMPT,
    ANSWER_PROMPT,
)
from reporter import NullReporter, ProgressReporter
from retrieval import Retriever
from skill_manager import SkillManager
from web_search import WebSearchClient

# ======================================================================
# 后向兼容导出 —— 新代码应直接从 services/ 引用
# ======================================================================

log = get_logger("auraderma.agent")

# ---------------------------------------------------------------------------
# Answer mode instruction templates
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
    weather_info: str = ""
    intent: str = "single"
    regimen_goal: str = ""
    regimen_plan: RegimenPlan | None = None
    multi_category_hits: dict[str, list] = field(default_factory=dict)
    workflow_plan: WorkflowPlan | None = None
    is_general_chat: bool = False


# ---------------------------------------------------------------------------
# Agent (lightweight orchestrator)
# ---------------------------------------------------------------------------


class SkincareAgent:
    """护肤品 AI 助手编排器。

    职责：
    - 持有所有服务的引用（意图、工作流、检索、回答等）
    - 编排处理流程：意图 → 工作流 → 检索 → 回答
    - 保持整体流程的可见性

    具体的业务逻辑已拆分到 services/ 模块中。
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        web: WebSearchClient,
        policy: MemoryPolicy,
        _embedder: object | None = None,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.web = web
        self.policy = policy
        self._embedder = _embedder
        self.reporter = reporter or NullReporter()

        # 延迟导入 service 以避免循环导入
        self._intent_service = None
        self._workflow_service = None
        self._regimen_service = None
        self._retrieval_service = None
        self._answer_service = None
        self._weather_service = None

    @property
    def _intent(self):
        if self._intent_service is None:
            from services import IntentService
            self._intent_service = IntentService(self.llm)
        return self._intent_service

    @property
    def _workflow(self):
        if self._workflow_service is None:
            from services import WorkflowService
            self._workflow_service = WorkflowService(self.llm)
        return self._workflow_service

    @property
    def _regimen(self):
        if self._regimen_service is None:
            from services import RegimenService
            self._regimen_service = RegimenService(self.llm)
        return self._regimen_service

    @property
    def _retrieval(self):
        if self._retrieval_service is None:
            from services.retrieval_service import RetrievalService
            self._retrieval_service = RetrievalService(self.llm, self.retriever, self._embedder)
        return self._retrieval_service

    @property
    def _answer(self):
        if self._answer_service is None:
            from services import AnswerService
            self._answer_service = AnswerService(self.llm)
        return self._answer_service

    @property
    def _weather(self):
        if self._weather_service is None:
            from services.weather_service import WeatherService
            self._weather_service = WeatherService(self.llm, None)
        return self._weather_service

    # ------------------------------------------------------------------
    # 公开 API（后向兼容）
    # ------------------------------------------------------------------

    def classify_intent(self, question: str) -> IntentResult:
        """[后向兼容] 意图分类委托给 IntentService。"""
        return self._intent.classify(question)

    def plan_regimen(self, question: str, goal: str) -> RegimenPlan:
        """[后向兼容] 护肤体系规划委托给 RegimenService。"""
        return self._regimen.plan(question, goal)

    def plan_workflows(self, question: str, intent: IntentResult) -> WorkflowPlan:
        """[后向兼容] 工作流规划委托给 WorkflowService。"""
        return self._workflow.plan(question, intent)

    # ------------------------------------------------------------------
    # 主路由（核心编排）
    # ------------------------------------------------------------------

    def route(
        self,
        ctx: AgentContext,
        memory_store: MemoryStore,
        skill_manager: SkillManager,
    ) -> RetrievalResult:
        """编排完整的处理流程。

        步骤：
        ① 意图分类 → ② 工作流规划 → ③ 执行流程（检索、技能、天气、网页搜索）
        """
        log.info(
            "开始处理: user=%s question_preview=%s...",
            ctx.user_id, ctx.question[:80],
        )

        # ① 意图分类
        self.reporter.intent()
        intent = self._intent.classify(ctx.question)

        # ② 工作流规划
        self.reporter.workflow()
        workflow = self._workflow.plan(ctx.question, intent)

        # ── 通用聊天模式 ──
        if "general_chat" in workflow.processes:
            return self._handle_general_chat(ctx, workflow, skill_manager)

        # ── 护肤相关流程 ──
        question_embedding = self._retrieval.embed_query(ctx.question)

        # ③ 通用检索
        memory_hits = self._retrieval.search_memory(question_embedding, ctx.user_id)
        doc_hits = self._retrieval.search_docs(question_embedding)

        memory_index_lines = memory_store.load_index(user_id=ctx.user_id, limit=20)
        index_recall_lines = memory_store.recall_index_lines(
            user_id=ctx.user_id, query=ctx.question, limit=6,
        )

        # ④ 记忆路由
        relevant_memory_ids, open_memory_files = self._handle_memory_routing(
            ctx, workflow, memory_index_lines, index_recall_lines, memory_store,
        )

        # ⑤ 产品检索
        retrieval_plan, product_hits, regimen_plan, multi_category_hits = (
            self._handle_product_search(ctx, intent, workflow, question_embedding)
        )

        # ⑥ 技能路由
        skill_names, skill_plan, skill_body = self._handle_skill_routing(
            ctx, intent, workflow, index_recall_lines, product_hits,
            memory_hits, doc_hits, skill_manager,
        )

        # ⑦ 天气
        weather_info = self._handle_weather(
            ctx, workflow, skill_names, memory_store,
        )

        # ⑧ 网页搜索
        web_notes = self._handle_web_search(ctx, workflow, product_hits,
                                             memory_hits, doc_hits,
                                             index_recall_lines, skill_names)

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
            memory_file_snippets=[],
            skill_body=skill_body,
            weather_info=weather_info,
            intent=intent.intent,
            regimen_goal=intent.goal,
            regimen_plan=regimen_plan,
            multi_category_hits=multi_category_hits,
            workflow_plan=workflow,
            is_general_chat=False,
        )

    def answer(
        self,
        ctx: AgentContext,
        memory_store: MemoryStore,
        skill_manager: SkillManager,
    ) -> str:
        """编排意图分类 → 检索 → 回答生成。"""
        route_result = self.route(ctx, memory_store, skill_manager)
        self.reporter.answer()
        return self._answer.answer(ctx, route_result)

    # ------------------------------------------------------------------
    # 分步处理
    # ------------------------------------------------------------------

    def _handle_general_chat(
        self,
        ctx: AgentContext,
        workflow,
        skill_manager,
    ) -> RetrievalResult:
        """处理纯聊天请求。"""
        weather_info = ""
        if "weather_check" in workflow.processes:
            profile_lines = ctx.memory.summary_lines(max_items_per_scope=20)
            ws = WeatherService(self.llm, skill_manager)
            weather_info = ws.fetch_weather(ctx.question, profile_lines)

        need_web = "web_search" in workflow.processes
        web_notes: list[str] = []
        if need_web and self.web.enabled:
            self.reporter.web_search(ctx.question)
            try:
                web_results = self.web.search(ctx.question, top_k=3)
                web_notes = [
                    f"网页搜索参考，仅供参考｜{r.title}｜{r.url}｜{r.snippet}"
                    for r in web_results
                ]
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
            intent="general",
            regimen_goal="",
            regimen_plan=None,
            multi_category_hits={},
            workflow_plan=workflow,
            is_general_chat=True,
        )

    def _handle_memory_routing(
        self,
        ctx,
        workflow,
        memory_index_lines,
        index_recall_lines,
        memory_store,
    ):
        """记忆路由处理。"""
        if "memory_lookup" not in workflow.processes:
            return [], False

        self.reporter.memory()
        relevant_ids, open_files = self._retrieval.route_memory(
            ctx.question, memory_index_lines, index_recall_lines,
        )

        memory_file_snippets = (
            memory_store.read_relevant_memory_texts(
                user_id=ctx.user_id,
                memory_ids=relevant_ids,
                query=ctx.question,
                limit=4,
            )
            if open_files
            else []
        )
        return relevant_ids, open_files

    def _handle_product_search(
        self,
        ctx,
        intent,
        workflow,
        question_embedding,
    ):
        """产品检索：按意图和检索模式分发。"""
        retrieval_plan: str = ""
        product_hits: list = []
        regimen_plan = None
        multi_category_hits: dict[str, list] = {}

        if "product_search" not in workflow.processes and "regimen_planning" not in workflow.processes:
            return retrieval_plan, product_hits, regimen_plan, multi_category_hits

        # 护肤体系模式
        if intent.intent == "regimen" or "regimen_planning" in workflow.processes:
            self.reporter.regimen()
            regimen_plan = self._regimen.plan(ctx.question, intent.goal)
            retrieval_plan, product_hits, _ = self._retrieval.search_regimen_products(
                ctx.question, regimen_plan,
            )
        # 多品类模式
        elif intent.intent == "multi" and len(intent.explicit_categories) >= 2:
            self.reporter.product_search("、".join(intent.explicit_categories))
            retrieval_plan, product_hits, multi_category_hits = (
                self._retrieval.search_multi_category(
                    ctx.question, intent.explicit_categories,
                )
            )
        # 单品模式
        else:
            self.reporter.product_search()
            retrieval_plan, product_hits = self._retrieval.search_single_product(
                ctx.question, question_embedding, intent.explicit_categories,
            )

        return retrieval_plan, product_hits, regimen_plan, multi_category_hits

    def _handle_skill_routing(
        self,
        ctx,
        intent,
        workflow,
        index_recall_lines,
        product_hits,
        memory_hits,
        doc_hits,
        skill_manager,
    ):
        """技能路由处理。"""
        skill_names: list[str] = []
        skill_plan = ""
        skill_body = ""

        need_skills = (
            "web_search" in workflow.processes
            or "weather_check" in workflow.processes
            or "file_read" in workflow.processes
        )
        if not need_skills:
            return skill_names, skill_plan, skill_body

        skill_names, skill_plan, skill_body = self._retrieval.route_skills(
            ctx.question, intent, index_recall_lines, index_recall_lines,
            product_hits, memory_hits, doc_hits, skill_manager,
        )
        return skill_names, skill_plan, skill_body

    def _handle_weather(self, ctx, workflow, skill_names, memory_store):
        """天气查询处理。"""
        if "weather_check" not in workflow.processes and "weather_check" not in skill_names:
            return ""

        self.reporter.thinking("城市识别")
        profile_lines = ctx.memory.summary_lines(max_items_per_scope=20)
        ws = WeatherService(self.llm, None)
        self._weather_service = ws
        return ws.fetch_weather(ctx.question, profile_lines)

    def _handle_web_search(
        self,
        ctx,
        workflow,
        product_hits,
        memory_hits,
        doc_hits,
        index_recall_lines,
        skill_names,
    ):
        """网页搜索处理。"""
        if "web_search" not in workflow.processes:
            return []

        need_web = RetrievalService.should_use_web(
            product_hits, memory_hits, doc_hits,
            index_recall_lines, skill_names,
        )
        if not need_web or not self.web.enabled:
            return []

        self.reporter.web_search(ctx.question)
        try:
            web_results = self.web.search(ctx.question, top_k=3)
            return [
                f"网页搜索参考，仅供参考｜{r.title}｜{r.url}｜{r.snippet}"
                for r in web_results
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 记忆提取和持久化
    # ------------------------------------------------------------------

    def auto_memory_extract(self, user_id: str, dialog_text: str) -> list:
        """自动从对话中提取记忆。"""
        prompt = (
            "请从以下对话中抽取可长期保存的护肤记忆，返回 JSON 数组。"
            "每项包含 text, scope(profile|long_term), confidence(0~1之间的数字，如0.8), tags。"
            "只返回 JSON，不要解释。\n\n"
            f"对话内容：\n{dialog_text}"
        )
        raw = self.llm.chat(SYSTEM_PROMPT, prompt, temperature=0.0)
        items = JsonParser.safe_parse_list(raw, context="memory_extract")

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

    def finalize_turn(
        self,
        user_id: str,
        dialog_text: str,
        memory_store: MemoryStore,
        memory_bundle: MemoryBundle,
    ) -> None:
        """完成一轮对话：自动提取记忆并持久化。"""
        extracted = self.auto_memory_extract(user_id=user_id, dialog_text=dialog_text)
        for item in extracted:
            memory_store.append(item)
            self._apply_memory_to_bundle(memory_bundle, item)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

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
    """格式化 JSON 输出（后向兼容导出）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)
