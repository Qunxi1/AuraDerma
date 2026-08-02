from __future__ import annotations

from dataclasses import asdict
from textwrap import dedent

from schema import (
    AgentContext,
    RetrievalResult,
    json_dumps_pretty,
)
from core import get_logger
from memory import aggregate_skin_profile

log = get_logger("auraderma.answer")

# ---------------------------------------------------------------------------
# Answer mode instruction templates (moved from agent.py)
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
        "5. One or two follow-up questions about your skin condition if appropriate.\n\n"
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


class AnswerService:
    """回答生成服务。

    根据检索结果和意图模式，生成最终用户回答。
    """

    def __init__(self, llm):
        self._llm = llm

    def answer(self, ctx: AgentContext, route: RetrievalResult) -> str:
        """生成回答，根据模式路由到不同策略。

        Args:
            ctx: 代理上下文
            route: 检索结果

        Returns:
            最终回答文本
        """
        if route.is_general_chat:
            return self._answer_general(ctx, route)

        log.info(
            "生成护肤回答: intent=%s regimen=%s multi=%s",
            route.intent,
            route.intent == "regimen" and route.regimen_plan is not None,
            route.intent == "multi" and bool(route.multi_category_hits),
        )

        if route.intent == "regimen" and route.regimen_plan:
            return self._answer_regimen(ctx, route)
        elif route.intent == "multi" and route.multi_category_hits:
            return self._answer_multi(ctx, route)
        else:
            return self._answer_single(ctx, route)

    def _answer_single(self, ctx: AgentContext, route: RetrievalResult) -> str:
        mode = "product_search"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        if route.workflow_plan and not route.workflow_plan.needs_product_search:
            mode = "skincare_analysis"
            mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        from prompts import ANSWER_PROMPT
        answer_prompt_filled = ANSWER_PROMPT.format(
            mode=mode, mode_instructions=mode_instructions,
        )

        context_block = self._build_context_block(ctx, route, mode, mode_instructions)
        return self._llm.chat(answer_prompt_filled, context_block)

    def _answer_multi(self, ctx: AgentContext, route: RetrievalResult) -> str:
        mode = "product_search"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        from prompts import ANSWER_PROMPT
        answer_prompt_filled = ANSWER_PROMPT.format(
            mode=mode, mode_instructions=mode_instructions,
        )

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
                    block += (
                        f"\n  · [{brand}] {name}{price_str}"
                        f" — {concerns}  [score={h.score:.2f}]"
                    )
            else:
                block += "\n  (当前知识库暂无该品类产品)"
            category_blocks.append(block)

        history_section = (
            f"对话历史:\n{route.history_block}" if route.history_block else ""
        )

        skin_profile_section = AnswerService._build_skin_profile_block(ctx)

        context_block = dedent(f"""
            用户问题:
            {ctx.question}

            意图模式: multi
            用户指定品类: {'、'.join(route.multi_category_hits.keys())}

            {history_section}

            {skin_profile_section}

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
            说明每个产品的品牌、价格，以及为什么适合用户（结合肤质画像）。
            如果某品类内部无产品，明确说"当前知识库暂无该品类产品"，不要编造。
            最后可以给出搭配建议和使用先后顺序（如有需要）。
        """).strip()
        return self._llm.chat(answer_prompt_filled, context_block)

    def _answer_general(self, ctx: AgentContext, route: RetrievalResult) -> str:
        mode = "general_chat"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        from prompts import ANSWER_PROMPT
        answer_prompt_filled = ANSWER_PROMPT.format(
            mode=mode, mode_instructions=mode_instructions,
        )

        weather_section = (
            f"当地气候数据:\n{route.weather_info}" if route.weather_info else ""
        )
        web_section = (
            f"网页搜索参考:\n{chr(10).join(route.web_notes)}" if route.web_notes else ""
        )
        history_section = (
            f"对话历史:\n{route.history_block}" if route.history_block else ""
        )

        context_block = dedent(f"""
            用户问题:
            {ctx.question}

            {history_section}

            这是一次纯聊天对话。不要涉及任何护肤、美容、产品推荐相关的内容。
            请直接用中文回答用户的问题，保持友好和帮助性。

            {weather_section}
            {web_section}
        """).strip()
        return self._llm.chat(answer_prompt_filled, context_block)

    def _answer_regimen(self, ctx: AgentContext, route: RetrievalResult) -> str:
        plan = route.regimen_plan
        assert plan is not None

        mode = "regimen_planning"
        mode_instructions = MODE_INSTRUCTIONS.get(mode, DEFAULT_INSTRUCTIONS)

        from prompts import ANSWER_PROMPT
        answer_prompt_filled = ANSWER_PROMPT.format(
            mode=mode, mode_instructions=mode_instructions,
        )

        regimen_blocks: list[str] = []

        def _fmt_group(label: str, steps) -> str:
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
                        lines.append(
                            f"       score={h.score:.2f} | [{brand}] {name}{price_str}"
                            f" | {concerns}",
                        )
                else:
                    lines.append("     (内部暂无匹配产品，建议网页搜索补充)")
            return "\n".join(lines)

        regimen_blocks.append(_fmt_group("☀️ 日间护理", plan.morning_steps))
        regimen_blocks.append(_fmt_group("🌙 夜间护理", plan.evening_steps))
        regimen_blocks.append(_fmt_group("📅 周期护理", plan.periodic_steps))

        must_have_str = "、".join(plan.must_have_categories) if plan.must_have_categories else "无"
        avoid_str = "、".join(plan.avoid_ingredients) if plan.avoid_ingredients else "无"

        history_section = (
            f"对话历史:\n{route.history_block}" if route.history_block else ""
        )

        skin_profile_section = AnswerService._build_skin_profile_block(ctx)

        context_block = dedent(f"""
            用户问题:
            {ctx.question}

            意图模式: regimen | 护肤目标: {plan.goal}
            目标说明: {plan.goal_explanation}

            {history_section}

            {skin_profile_section}

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
            所有推荐必须结合用户的肤质画像（肤质类型、T区/脸颊状态、敏感程度、护肤诉求），
            解释为什么该产品适合用户的特定皮肤状况。
            最后给出完整的使用流程总结和注意事项。
        """).strip()
        return self._llm.chat(answer_prompt_filled, context_block)

    @staticmethod
    def _build_skin_profile_block(ctx: AgentContext) -> str:
        """构建结构化的用户肤质画像板块（从 profile memories 聚合）。"""
        profile = aggregate_skin_profile(ctx.memory.profile)
        block = profile.to_formatted_block()
        if block:
            return block
        # 如果 profile 为空，试试从 short_term 和 long_term 中找肤质信息
        all_skin = [
            m for m in ctx.memory.short_term + ctx.memory.long_term
            if any(kw in m.text for kw in ["肤质", "油", "干", "敏感", "痘", "过敏"])
        ]
        if all_skin:
            profile = aggregate_skin_profile(all_skin)
            return profile.to_formatted_block()
        return ""

    @staticmethod
    def _build_context_block(
        ctx: AgentContext,
        route: RetrievalResult,
        mode: str,
        mode_instructions: str,
    ) -> str:
        """构建注入 LLM 的完整 context block，包含对话历史、检索结果、肤质画像等。

        类似 Reasonix 的 Compose：system prompt 保持稳定，history 和 memory 追加在 turn tail。
        """
        from textwrap import dedent

        history_section = (
            f"对话历史:\n{route.history_block}" if route.history_block else ""
        )

        skin_profile_section = AnswerService._build_skin_profile_block(ctx)

        return dedent(f"""
            用户问题:
            {ctx.question}

            {history_section}

            意图模式: {route.intent} | 护肤目标: {route.regimen_goal}

            检索计划:
            {route.retrieval_plan}

            工作流规划:
            {json_dumps_pretty(asdict(route.workflow_plan)) if route.workflow_plan else '无'}

            {skin_profile_section}

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
            {AnswerService._format_hits(route.product_hits)}

            记忆召回:
            {AnswerService._format_hits(route.memory_hits)}

            文档召回:
            {AnswerService._format_hits(route.doc_hits)}

            网页搜索参考:
            {chr(10).join(route.web_notes) if route.web_notes else '无'}

            当地气候数据:
            {route.weather_info if route.weather_info else '无'}
        """).strip()

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
