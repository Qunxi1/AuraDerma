"""ReAct Agent —— 基于思考-行动-观察循环的护肤品 AI 顾问。

与 Pipeline Agent 的区别:
- Pipeline: 意图分类→工作流规划→检索→回答 (单程直线)
- ReAct:    思考→调用工具→观察结果→再思考→...→最终回答 (迭代循环)

优势:
- 每一步的决策都基于前一步的实际结果
- LLM 可以根据检索结果动态调整策略
- 更智能、更灵活
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from core.json_parser import JsonParser
from core.logger import get_logger
from llm import LLMClient
from memory import MemoryBundle, MemoryItem, MemoryScope, MemoryStore, aggregate_skin_profile, new_id
from prompts import SYSTEM_PROMPT
from prompts_react import REACT_SYSTEM_PROMPT, REACT_TOOLS

log = get_logger("auraderma.react")


@dataclass(slots=True)
class ReActResult:
    """一次 ReAct 循环的结果。"""
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_tool_calls: int = 0


class ReActAgent:
    """基于 ReAct 模式的护肤品 AI 顾问。

    使用方式:
    1. 初始化时注入 LLM、检索器、搜索引擎等依赖。
    2. 调用 run() 执行一次完整的"思考→行动→观察"循环。
    """

    MAX_STEPS = 8

    def __init__(
        self,
        llm: LLMClient,
        retriever,       # Retriever (Qdrant)
        embedder,        # LocalEmbedder
        web,             # WebSearchClient
        reporter=None,   # ProgressReporter
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.embedder = embedder
        self.web = web
        self.reporter = reporter

        self._cached_profile_text: str = ""
        self._cached_user_id: str = ""

    # ==================================================================
    # 主入口
    # ==================================================================

    def run(
        self,
        question: str,
        user_id: str,
        memory_store: MemoryStore,
    ) -> ReActResult:
        """执行一次完整的 ReAct 循环。

        Returns:
            ReActResult (包含最终回答和执行步骤)
        """
        memory = memory_store.load(user_id)
        profile_text = self._build_profile_text(user_id, memory)
        system_prompt = self._build_system_prompt(profile_text)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        steps: list[dict[str, Any]] = []
        tool_call_count = 0
        final_answer = ""

        for step_idx in range(self.MAX_STEPS):
            if self.reporter:
                self.reporter.thinking(f"ReAct 第 {step_idx + 1} 轮思考")

            response = self.llm.chat_messages(
                messages=messages,
                temperature=0.3,
                tools=REACT_TOOLS,
                tool_choice="auto",
            )

            content = response.get("content", "")
            tool_calls = response.get("tool_calls")

            # 情况1: LLM 想调用工具
            if tool_calls:
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)

                for tc in tool_calls:
                    tool_name = tc["name"]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    tool_result = self._execute_tool(tool_name, args, user_id, memory)
                    tool_call_count += 1

                    steps.append({
                        "step": step_idx + 1,
                        "tool": tool_name,
                        "args": args,
                        "result_preview": tool_result[:200],
                    })
                    log.info(
                        "ReAct step=%d tool=%s args=%s result=%s...",
                        step_idx + 1, tool_name, args, tool_result[:80],
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            # 情况2: LLM 给出最终回答
            elif content and content.strip():
                final_answer = content.strip()
                log.info("ReAct 完成: steps=%d answer_len=%d", tool_call_count, len(final_answer))
                break

            # 情况3: 既没有 tool_calls 也没有 content
            else:
                log.warning("ReAct step=%d 空响应", step_idx + 1)
                if tool_call_count > 0:
                    messages.append({
                        "role": "user",
                        "content": "请基于现有的信息，直接给出最终回答。不要继续调用工具。",
                    })
                else:
                    final_answer = "抱歉，我暂时无法处理你的问题，请稍后再试。"
                    break

        # 循环结束仍未得到答案，强制要求回答
        if not final_answer:
            messages.append({
                "role": "user",
                "content": "你已经调用了所有可用工具。请基于以上所有信息，直接给出完整的最终回答。",
            })
            response = self.llm.chat_messages(messages=messages, temperature=0.3)
            final_answer = response.get("content", "抱歉，暂时无法回答你的问题。").strip()

        self._save_to_memory(user_id, question, final_answer, memory_store)
        return ReActResult(answer=final_answer, steps=steps, total_tool_calls=tool_call_count)

    # ==================================================================
    # 工具分发
    # ==================================================================

    def _execute_tool(
        self, tool_name: str, args: dict[str, Any], user_id: str, memory: MemoryBundle,
    ) -> str:
        dispatcher: dict[str, Callable[[dict, str, MemoryBundle], str]] = {
            "search_products": self._tool_search_products,
            "search_memory": self._tool_search_memory,
            "lookup_profile": self._tool_lookup_profile,
            "web_search": self._tool_web_search,
            "weather_check": self._tool_weather_check,
            "search_docs": self._tool_search_docs,
        }
        handler = dispatcher.get(tool_name)
        if handler is None:
            return f"未知工具: {tool_name}。可用: {', '.join(dispatcher.keys())}"
        try:
            return handler(args, user_id, memory)
        except Exception as exc:
            log.exception("工具执行失败: tool=%s", tool_name)
            return f"工具 {tool_name} 执行失败: {exc}"

    # ── search_products ──

    def _tool_search_products(self, args, user_id, memory) -> str:
        query = args.get("query", "")
        limit = min(args.get("limit", 5), 10)
        if not query:
            return "错误: 缺少 query 参数"

        embedding = self.embedder.embed([query])[0]
        hits = self.retriever.search(self.retriever.products_collection, embedding, limit=limit)

        if not hits:
            return f"知识库中未找到与「{query}」匹配的产品。"

        lines = [f"产品搜索「{query}」返回 {len(hits)} 个结果:"]
        for i, h in enumerate(hits, 1):
            p = h.payload
            name = p.get("name", "未知")
            brand = f"[{p['brand']}] " if p.get("brand") else ""
            price = f" ¥{p['price_cny']}" if p.get("price_cny") else ""
            category = p.get("category", "")
            efficacy = (p.get("core_efficacy") or p.get("efficacy", ""))[:100]
            skin_types = p.get("skin_types", [])
            concerns = p.get("concerns", [])
            ingredients = p.get("ingredients", [])[:8]

            lines.append(
                f"\n{i}. {brand}**{name}**{price} | {category}"
                f"\n   功效: {efficacy}"
                f"\n   适合肤质: {', '.join(skin_types) if skin_types else '未标注'}"
                f"\n   针对问题: {', '.join(concerns[:5]) if concerns else '未标注'}"
                f"\n   主要成分: {', '.join(ingredients) if ingredients else '未标注'}"
            )
        return "\n".join(lines)

    # ── search_memory ──

    def _tool_search_memory(self, args, user_id, memory) -> str:
        query = args.get("query", "")
        if not query:
            return "错误: 缺少 query 参数"

        embedding = self.embedder.embed([query])[0]
        hits = self.retriever.search(self.retriever.memory_collection, embedding, limit=5)

        if not hits:
            return f"未在记忆中搜索到与「{query}」相关的内容。"

        lines = [f"记忆搜索「{query}」结果:"]
        for h in hits:
            scope = h.payload.get("scope", "?")
            text = (h.payload.get("text") or h.payload.get("summary", ""))[:150]
            lines.append(f"- [{scope}] {text}")
        return "\n".join(lines)

    # ── lookup_profile ──

    def _tool_lookup_profile(self, args, user_id, memory) -> str:
        profile = aggregate_skin_profile(memory.profile)
        if profile.is_empty():
            all_skin = [
                m for m in memory.short_term + memory.long_term
                if any(kw in m.text.lower() for kw in
                       ["肤质", "油", "干", "敏感", "痘", "过敏", "t区", "脸颊", "泛红"])
            ]
            if all_skin:
                profile = aggregate_skin_profile(all_skin)

        if profile.is_empty():
            return (
                "用户尚未提供肤质信息。\n"
                "建议向用户询问: ① T区/脸颊的油干状况 ② 是否容易过敏 ③ 是否长痘 ④ 护肤目标"
            )

        block = profile.to_formatted_block()
        if profile.raw_memories:
            block += "\n\n原始记忆:\n" + "\n".join(f"  - {t[:100]}" for t in profile.raw_memories[:10])
        return block

    # ── web_search ──

    def _tool_web_search(self, args, user_id, memory) -> str:
        query = args.get("query", "")
        if not query:
            return "错误: 缺少 query 参数"

        if not self.web or not self.web.enabled:
            return "网页搜索当前不可用。"

        try:
            results = self.web.search(query, top_k=3)
            if not results:
                return f"网页搜索「{query}」无结果。"

            lines = [f"网页搜索「{query}」结果 (网页搜索参考，仅供参考):"]
            for r in results:
                lines.append(f"- **{r.title}**\n  摘要: {r.snippet[:200]}\n  链接: {r.url}")
            return "\n".join(lines)
        except Exception as exc:
            return f"网页搜索失败: {exc}"

    # ── weather_check ──

    def _tool_weather_check(self, args, user_id, memory) -> str:
        city = args.get("city", "")
        if not city:
            return "错误: 缺少 city 参数"

        import urllib.parse
        import urllib.request

        encoded = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded}?format=j1"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return f"天气查询失败: {exc}"

        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        aname = (area.get("areaName", [{}]) or [{}])[0].get("value", city) if area.get("areaName") else city
        temp = current.get("temp_C", "?")
        humidity = current.get("humidity", "?")
        cond = (current.get("weatherDesc", [{}]) or [{}])[0].get("value", "未知")
        feels = current.get("FeelsLikeC", "?")

        return f"城市: {aname} | 温度: {temp}°C (体感 {feels}°C) | 湿度: {humidity}% | 天气: {cond}"

    # ── search_docs ──

    def _tool_search_docs(self, args, user_id, memory) -> str:
        query = args.get("query", "")
        if not query:
            return "错误: 缺少 query 参数"

        embedding = self.embedder.embed([query])[0]
        hits = self.retriever.search(self.retriever.docs_collection, embedding, limit=4)

        if not hits:
            return f"文档知识库中未找到与「{query}」相关的内容。"

        lines = [f"文档搜索「{query}」结果:"]
        for h in hits:
            title = (h.payload.get("title") or h.payload.get("section") or "")
            text = (h.payload.get("text", ""))[:200]
            lines.append(f"- **{title}**\n  {text}...")
        return "\n".join(lines)

    # ==================================================================
    # 辅助方法
    # ==================================================================

    def _build_system_prompt(self, profile_text: str) -> str:
        if profile_text:
            return REACT_SYSTEM_PROMPT + "\n\n" + profile_text
        return REACT_SYSTEM_PROMPT

    def _build_profile_text(self, user_id: str, memory: MemoryBundle) -> str:
        if self._cached_user_id == user_id and self._cached_profile_text:
            return self._cached_profile_text

        profile = aggregate_skin_profile(memory.profile)
        if profile.is_empty():
            all_skin = [
                m for m in memory.short_term + memory.long_term
                if any(kw in m.text.lower() for kw in
                       ["肤质", "油", "干", "敏感", "痘", "过敏", "t区", "脸颊", "泛红"])
            ]
            if all_skin:
                profile = aggregate_skin_profile(all_skin)

        self._cached_user_id = user_id
        self._cached_profile_text = profile.to_formatted_block() if not profile.is_empty() else ""
        return self._cached_profile_text

    def _save_to_memory(
        self, user_id: str, question: str, answer: str, memory_store: MemoryStore,
    ) -> None:
        dialog_text = f"用户: {question}\n助手: {answer}"
        prompt = (
            "请从以下对话中抽取可长期保存的护肤记忆，返回 JSON 数组。"
            "每项包含 text, scope(profile|long_term), confidence(0~1), tags。"
            "只返回 JSON，不要解释。\n\n"
            f"对话内容：\n{dialog_text}"
        )
        try:
            raw = self.llm.chat(SYSTEM_PROMPT, prompt, temperature=0.0)
            items = JsonParser.safe_parse_list(raw, context="react_memory_extract")
            for item in items:
                scope = item.get("scope", "long_term")
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                mem = MemoryItem(
                    memory_id=new_id(),
                    scope=MemoryScope(scope),
                    user_id=user_id,
                    text=text,
                    summary=text[:120],
                    confidence=float(item.get("confidence", 0.7)),
                    ttl_days=None if scope == "profile" else 365,
                    tags=item.get("tags", []),
                )
                if scope == "profile":
                    mem.ttl_days = None
                    if "profile" not in mem.tags:
                        mem.tags.append("profile")
                memory_store.append(mem)
        except Exception as exc:
            log.warning("ReAct 记忆提取失败: %s", exc)
        self._cached_user_id = ""
