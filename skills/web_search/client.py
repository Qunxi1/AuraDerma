"""Web 搜索客户端 —— 支持 10 种搜索引擎。

API 设计参考 DeepSeek-Reasonix 的 ``src/tools/web.ts``。

所有 API 密钥统一在 ``~/.auraderma/search_config.json`` 中管理，
也可通过同名环境变量覆盖。Bing（cn.bing.com HTML 爬取）为默认，
无需任何 API Key。

支持引擎：
  - ``bing``       — HTML 爬取 cn.bing.com（默认，国内可用，无需 Key）
  - ``bing-intl``  — HTML 爬取 www.bing.com（国际版）
  - ``searxng``    — 自托管 SearXNG 实例
  - ``metaso``     — 秘塔 AI 搜索
  - ``baidu``      — 百度 AI 搜索（千帆）
  - ``tavily``     — Tavily Search API
  - ``perplexity`` — Perplexity AI
  - ``exa``        — Exa API
  - ``brave``      — Brave Search API
  - ``ollama``     — Ollama Cloud Web Search
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from search_config import SearchApiKeys, save_search_config


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source_note: str = "网页搜索参考，仅供参考"


# ---------------------------------------------------------------------------
# 引擎元数据注册表
# ---------------------------------------------------------------------------

SEARCH_PROVIDERS: dict[str, dict] = {
    "bing": {
        "label": "Bing（cn.bing.com，HTML 爬取，无需 Key，国内可用）",
        "needs_api_key": False,
        "description": "通过 HTML 爬取 cn.bing.com 搜索结果，无需任何 API Key",
    },
    "bing-intl": {
        "label": "Bing International（www.bing.com，HTML 爬取，无需 Key）",
        "needs_api_key": False,
        "description": "通过 HTML 爬取 www.bing.com 搜索结果（国际版）",
    },
    "searxng": {
        "label": "SearXNG（自托管元搜索引擎，无需 Key）",
        "needs_api_key": False,
        "description": "自托管 SearXNG 实例，需在 search_config.json 中配置 searxngEndpoint",
    },
    "metaso": {
        "label": "秘塔 AI 搜索（需 METASO_API_KEY）",
        "needs_api_key": True,
        "key_field": "metaso",
        "description": "调用秘塔 AI 搜索 API，需在 search_config.json 中配置 metasoApiKey",
    },
    "baidu": {
        "label": "百度 AI 搜索 / 千帆（需 BAIDU_API_KEY）",
        "needs_api_key": True,
        "key_field": "baidu",
        "description": "调用百度千帆 AI 搜索 API，需在 search_config.json 中配置 baiduApiKey",
    },
    "tavily": {
        "label": "Tavily Search（需 TAVILY_API_KEY，结构化 JSON）",
        "needs_api_key": True,
        "key_field": "tavily",
        "description": "调用 Tavily Search API，返回结构化 JSON 结果",
    },
    "perplexity": {
        "label": "Perplexity AI（需 PERPLEXITY_API_KEY）",
        "needs_api_key": True,
        "key_field": "perplexity",
        "description": "调用 Perplexity AI 搜索 API",
    },
    "exa": {
        "label": "Exa API（需 EXA_API_KEY）",
        "needs_api_key": True,
        "key_field": "exa",
        "description": "调用 Exa API 进行 AI 驱动搜索",
    },
    "brave": {
        "label": "Brave Search API（需 BRAVE_API_KEY）",
        "needs_api_key": True,
        "key_field": "brave",
        "description": "调用 Brave Search API，免费 2000 次/月",
    },
    "ollama": {
        "label": "Ollama Cloud Web Search（需 OLLAMA_API_KEY）",
        "needs_api_key": True,
        "key_field": "ollama",
        "description": "调用 Ollama Cloud 的 Web Search API",
    },
}


# ---------------------------------------------------------------------------
# 端点常量
# ---------------------------------------------------------------------------

BING_ENDPOINT = "https://cn.bing.com/search"
BING_INTL_ENDPOINT = "https://www.bing.com/search"
METASO_ENDPOINT = "https://metaso.cn/api/v1"
BAIDU_AI_SEARCH_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/web_search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"
EXA_ENDPOINT = "https://api.exa.ai/answer"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
OLLAMA_WEB_SEARCH_ENDPOINT = "https://ollama.com/api/web_search"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SEARCH_TIMEOUT_S = 15


# ---------------------------------------------------------------------------
# WebSearchClient
# ---------------------------------------------------------------------------

class WebSearchClient:
    """Web 搜索客户端，支持 10 个搜索引擎。

    默认使用 ``bing``（cn.bing.com HTML 爬取，无需 API Key）。
    可通过 ``set_provider()`` 运行时切换，或修改配置文件中的默认值。

    API 密钥统一在 ``~/.auraderma/search_config.json`` 中管理。
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.provider = self._resolve_default_engine()

    @staticmethod
    def _resolve_default_engine() -> str:
        engine = SearchApiKeys.default_engine()
        return engine if engine in SEARCH_PROVIDERS else "bing"

    # ------------------------------------------------------------------
    # 提供方信息
    # ------------------------------------------------------------------

    @staticmethod
    def list_providers() -> dict[str, str]:
        """返回所有可用提供方名称 → 标签的映射。"""
        return {k: v["label"] for k, v in SEARCH_PROVIDERS.items()}

    @property
    def current_provider_label(self) -> str:
        info = SEARCH_PROVIDERS.get(self.provider)
        return info["label"] if info else self.provider

    # ------------------------------------------------------------------
    # 运行时切换
    # ------------------------------------------------------------------

    def set_provider(self, provider: str) -> None:
        provider = provider.lower().strip()
        if provider not in SEARCH_PROVIDERS:
            raise ValueError(
                f"不支持的搜索提供方：'{provider}'。"
                f"可用选项：{', '.join(SEARCH_PROVIDERS)}"
            )
        self.provider = provider

    # ------------------------------------------------------------------
    # 统一搜索入口
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if not self.enabled:
            return []
        if self.provider == "bing":
            return self._search_bing(query, top_k, BING_ENDPOINT)
        if self.provider == "bing-intl":
            return self._search_bing(query, top_k, BING_INTL_ENDPOINT)
        if self.provider == "searxng":
            return self._search_searxng(query, top_k)
        if self.provider == "metaso":
            return self._search_metaso(query, top_k)
        if self.provider == "baidu":
            return self._search_baidu(query, top_k)
        if self.provider == "tavily":
            return self._search_tavily(query, top_k)
        if self.provider == "perplexity":
            return self._search_perplexity(query, top_k)
        if self.provider == "exa":
            return self._search_exa(query, top_k)
        if self.provider == "brave":
            return self._search_brave(query, top_k)
        if self.provider == "ollama":
            return self._search_ollama(query, top_k)
        raise NotImplementedError(f"不支持的搜索提供方：{self.provider}")

    # ==================================================================
    # 1. Bing — HTML 爬取
    # ==================================================================

    def _search_bing(self, query: str, top_k: int, endpoint: str) -> list[WebSearchResult]:
        """通过 HTML 爬取 Bing 搜索结果。

        参考 Reasonix 的 ``searchBing()`` 和 ``parseBingResults()``。
        """
        top_k = max(1, min(10, top_k))
        try:
            resp = requests.get(
                endpoint,
                params={"q": query},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                timeout=SEARCH_TIMEOUT_S,
                allow_redirects=True,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Bing 搜索请求失败：{e}") from e

        html = resp.text
        results = self._parse_bing_results(html)[:top_k]

        if not results:
            if re.search(r"captcha|verify you are human|access denied|forbidden", html, re.IGNORECASE):
                raise RuntimeError(
                    "Bing 反爬页面，可能被频率限制。"
                    "可稍后再试，或切换到其他搜索引擎。"
                )
            return []

        return results

    @staticmethod
    def _parse_bing_results(html: str) -> list[WebSearchResult]:
        """解析 Bing HTML 搜索结果。选择器：li.b_algo > h2 a + div.b_caption p。"""
        soup = BeautifulSoup(html, "html.parser")
        results: list[WebSearchResult] = []

        for li in soup.select("li.b_algo"):
            anchor = li.select_one("h2 a[href]")
            if not anchor:
                continue
            href = anchor.get("href", "")
            if not href:
                continue
            title = anchor.get_text(strip=True)
            if not title:
                continue

            cap = li.select_one("div.b_caption p")
            snippet = cap.get_text(strip=True) if cap else ""
            snippet = re.sub(r"\s+", " ", snippet).strip()

            results.append(WebSearchResult(title=title, url=href, snippet=snippet))

        return results

    # ==================================================================
    # 2. SearXNG — 自托管 HTML 爬取
    # ==================================================================

    def _search_searxng(self, query: str, top_k: int) -> list[WebSearchResult]:
        """通过 SearXNG 实例执行搜索。"""
        top_k = max(1, min(10, top_k))
        endpoint = SearchApiKeys.searxng_endpoint() or "http://localhost:8080"

        try:
            resp = requests.get(
                f"{endpoint.rstrip('/')}/search",
                params={"format": "html", "q": query},
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html",
                },
                timeout=SEARCH_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"SearXNG 请求失败（端点：{endpoint}）：{e}") from e

        html = resp.text
        results = self._parse_searxng_results(html)[:top_k]
        return results

    @staticmethod
    def _parse_searxng_results(html: str) -> list[WebSearchResult]:
        """解析 SearXNG HTML 结果。"""
        soup = BeautifulSoup(html, "html.parser")
        results: list[WebSearchResult] = []

        # 优先尝试 article.result
        for article in soup.select("article.result, div.result"):
            link = article.select_one("h3 a, h4 a, a[href^='http']")
            if not link:
                continue
            href = link.get("href", "")
            if not href:
                continue
            title = link.get_text(strip=True)
            if not title:
                continue

            snippet = ""
            for p in article.select("p"):
                text = p.get_text(strip=True)
                if len(text) > 10 and text not in title:
                    snippet = text
                    break
            if not snippet:
                cs = article.select_one(".content, .result-content, [class*='snippet']")
                if cs:
                    snippet = cs.get_text(strip=True)

            results.append(WebSearchResult(title=title, url=href, snippet=snippet))

        # 回退：h3 > a 直接提取
        if not results:
            for a in soup.select("h3 a[href]"):
                href = a.get("href", "")
                if not href or href.startswith("#"):
                    continue
                title = a.get_text(strip=True)
                if not title:
                    continue
                snippet = ""
                p = a.find_parent().find_next_sibling("p") if a.find_parent() else None
                if p:
                    snippet = p.get_text(strip=True)
                results.append(WebSearchResult(title=title, url=href, snippet=snippet))

        return results

    # ==================================================================
    # 3. Metaso — REST API
    # ==================================================================

    def _search_metaso(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用秘塔 AI 搜索 API。"""
        api_key = SearchApiKeys.metaso()
        if not api_key:
            raise RuntimeError(
                "Metaso 搜索需要配置 metasoApiKey。\n"
                "请编辑 ~/.auraderma/search_config.json 添加 metasoApiKey 字段。"
            )
        top_k = max(1, min(100, top_k))

        try:
            resp = requests.post(
                f"{METASO_ENDPOINT}/search",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={"q": query, "scope": "webpage", "size": top_k},
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Metaso 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Metaso API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Metaso 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Metaso 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Metaso 返回格式异常：{e}") from e

        # 错误码处理
        code = data.get("code", 0)
        if code == 3003:
            raise RuntimeError("Metaso 每日调用额度已用完。")
        if code == 2005:
            raise RuntimeError("Metaso API Key 无效或被拒绝。")
        if code and code != 0:
            raise RuntimeError(f"Metaso API 错误 (code={code}): {data.get('message', '')}")

        results = []
        for wp in data.get("webpages", [])[:top_k]:
            results.append(
                WebSearchResult(
                    title=wp.get("title", ""),
                    url=wp.get("link", ""),
                    snippet=wp.get("snippet") or wp.get("summary", ""),
                )
            )
        return results

    # ==================================================================
    # 4. Baidu — 百度千帆 AI 搜索
    # ==================================================================

    def _search_baidu(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用百度千帆 AI 搜索 API。"""
        api_key = SearchApiKeys.baidu()
        if not api_key:
            raise RuntimeError(
                "百度 AI 搜索需要配置 baiduApiKey。\n"
                "请编辑 ~/.auraderma/search_config.json 添加 baiduApiKey 字段。"
            )
        top_k = max(1, min(10, top_k))

        try:
            resp = requests.post(
                BAIDU_AI_SEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"messages": [{"role": "user", "content": query}]},
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"百度 AI 搜索请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("百度 API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("百度请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"百度服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json() if resp.text else {}
        except ValueError as e:
            raise RuntimeError(f"百度返回格式异常：{e}") from e

        results = []
        for ref in data.get("references", [])[:top_k]:
            title = ref.get("title", "")
            url = ref.get("url", "")
            if not title or not url:
                continue
            snippet = ref.get("content") or ref.get("snippet", "")
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))

        return results

    # ==================================================================
    # 5. Tavily — REST API
    # ==================================================================

    def _search_tavily(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用 Tavily Search API。（参考 Reasonix searchTavily()）"""
        api_key = SearchApiKeys.tavily()
        if not api_key:
            raise RuntimeError(
                "Tavily 搜索需要配置 tavilyApiKey。\n"
                "前往 https://app.tavily.com 获取 Key，"
                "然后编辑 ~/.auraderma/search_config.json 添加 tavilyApiKey。"
            )
        top_k = max(1, min(20, top_k))

        try:
            resp = requests.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": top_k,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                },
                timeout=20,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Tavily 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Tavily API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Tavily 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Tavily 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Tavily 返回格式异常：{e}") from e

        results = []
        for row in data.get("results", [])[:top_k]:
            results.append(
                WebSearchResult(
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    snippet=row.get("content", ""),
                )
            )
        return results

    # ==================================================================
    # 6. Perplexity — AI Chat API
    # ==================================================================

    def _search_perplexity(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用 Perplexity AI 搜索。（参考 Reasonix searchPerplexity()）"""
        api_key = SearchApiKeys.perplexity()
        if not api_key:
            raise RuntimeError(
                "Perplexity 搜索需要配置 perplexityApiKey。\n"
                "前往 https://perplexity.ai/settings/api 获取 Key，"
                "然后编辑 ~/.auraderma/search_config.json。"
            )
        top_k = max(1, min(20, top_k))

        try:
            resp = requests.post(
                PERPLEXITY_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1024,
                    "return_related_questions": False,
                },
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Perplexity 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Perplexity API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Perplexity 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Perplexity 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Perplexity 返回格式异常：{e}") from e

        answer = ""
        choices = data.get("choices", [])
        if choices and choices[0].get("message", {}).get("content"):
            answer = choices[0]["message"]["content"]

        citations = data.get("citations", []) if isinstance(data.get("citations"), list) else []

        results: list[WebSearchResult] = []

        # 第一条放 AI 生成的 answer
        if answer:
            results.append(WebSearchResult(title="Perplexity AI 回答", url="", snippet=answer))

        count = min(len(citations), top_k)
        for i in range(count):
            c = citations[i]
            if isinstance(c, str):
                results.append(WebSearchResult(title=f"来源 {i + 1}", url=c, snippet=""))
            elif isinstance(c, dict) and isinstance(c.get("url"), str):
                results.append(
                    WebSearchResult(
                        title=c.get("title", f"来源 {i + 1}"),
                        url=c["url"],
                        snippet=str(c.get("text", "")),
                    )
                )

        return results

    # ==================================================================
    # 7. Exa — Answer API
    # ==================================================================

    def _search_exa(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用 Exa API。（参考 Reasonix searchExa()）"""
        api_key = SearchApiKeys.exa()
        if not api_key:
            raise RuntimeError(
                "Exa 搜索需要配置 exaApiKey。\n"
                "前往 https://exa.ai 注册获取 Key（免费 1000 次/月），"
                "然后编辑 ~/.auraderma/search_config.json。"
            )
        top_k = max(1, min(20, top_k))

        try:
            resp = requests.post(
                EXA_ENDPOINT,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={"query": query, "text": True},
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Exa 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Exa API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Exa 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Exa 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Exa 返回格式异常：{e}") from e

        answer = data.get("answer", "")
        citations = data.get("citations", []) if isinstance(data.get("citations"), list) else []

        results: list[WebSearchResult] = []

        # 第一条放 AI answer
        if answer:
            results.append(WebSearchResult(title="Exa AI 回答", url="", snippet=answer))

        for c in citations[:top_k]:
            url = c.get("url", "")
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=c.get("title", ""),
                    url=url,
                    snippet=c.get("text", ""),
                )
            )

        return results

    # ==================================================================
    # 8. Brave — Search API
    # ==================================================================

    def _search_brave(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用 Brave Search API。（参考 Reasonix searchBrave()）"""
        api_key = SearchApiKeys.brave()
        if not api_key:
            raise RuntimeError(
                "Brave 搜索需要配置 braveApiKey。\n"
                "前往 https://brave.com/search/api/ 注册获取 Key（免费 2000 次/月），"
                "然后编辑 ~/.auraderma/search_config.json。"
            )
        top_k = max(1, min(20, top_k))

        try:
            resp = requests.get(
                BRAVE_ENDPOINT,
                params={"q": query, "count": top_k},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Brave 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Brave API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Brave 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Brave 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Brave 返回格式异常：{e}") from e

        results = []
        for row in (data.get("web", {}) if isinstance(data.get("web"), dict) else {}).get("results", [])[:top_k]:
            results.append(
                WebSearchResult(
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    snippet=row.get("description", ""),
                )
            )
        return results

    # ==================================================================
    # 9. Ollama — Cloud Web Search
    # ==================================================================

    def _search_ollama(self, query: str, top_k: int) -> list[WebSearchResult]:
        """调用 Ollama Cloud 的 Web Search API。（参考 Reasonix searchOllama()）"""
        api_key = SearchApiKeys.ollama()
        if not api_key:
            raise RuntimeError(
                "Ollama Web Search 需要配置 ollamaApiKey。\n"
                "请编辑 ~/.auraderma/search_config.json 添加 ollamaApiKey 字段。"
            )
        top_k = max(1, min(10, top_k))

        try:
            resp = requests.post(
                OLLAMA_WEB_SEARCH_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"query": query, "max_results": top_k},
                timeout=SEARCH_TIMEOUT_S,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama 请求失败：{e}") from e

        if resp.status_code in (401, 403):
            raise RuntimeError("Ollama API Key 无效或被拒绝。")
        if resp.status_code == 429:
            raise RuntimeError("Ollama 请求频率超限。")
        if not resp.ok:
            raise RuntimeError(f"Ollama 服务器错误 (HTTP {resp.status_code})")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Ollama 返回格式异常：{e}") from e

        results = []
        for i, row in enumerate(data.get("results", [])[:top_k]):
            results.append(
                WebSearchResult(
                    title=row.get("title", "") or f"结果 {i + 1}",
                    url=row.get("url", ""),
                    snippet=row.get("content", ""),
                )
            )
        return results

    # ==================================================================
    # 通用 fetch
    # ==================================================================

    def fetch(self, url: str) -> str:
        """抓取指定 URL 的原始 HTML 文本。"""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError(f"抓取页面失败：{e}") from e
