from __future__ import annotations

from dataclasses import dataclass
import os
import requests


@dataclass(slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source_note: str = "网页搜索参考，仅供参考"


class WebSearchClient:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.provider = os.getenv("AURADERMA_WEB_SEARCH_PROVIDER", "tavily").lower()
        self.api_key = os.getenv("AURADERMA_WEB_SEARCH_API_KEY", "")
        self.base_url = os.getenv("AURADERMA_WEB_SEARCH_BASE_URL", "https://api.tavily.com")

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if not self.enabled:
            return []
        if self.provider == "tavily":
            return self._search_tavily(query, top_k)
        if self.provider == "bing":
            return self._search_bing(query, top_k)
        raise NotImplementedError(f"unsupported web search provider: {self.provider}")

    def _search_tavily(self, query: str, top_k: int) -> list[WebSearchResult]:
        if not self.api_key:
            raise RuntimeError("AURADERMA_WEB_SEARCH_API_KEY is required for Tavily")
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": False,
        }
        resp = requests.post(f"{self.base_url.rstrip('/')}/search", json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for row in data.get("results", [])[:top_k]:
            results.append(WebSearchResult(title=row.get("title", ""), url=row.get("url", ""), snippet=row.get("content", "")))
        return results

    def _search_bing(self, query: str, top_k: int) -> list[WebSearchResult]:
        raise NotImplementedError("Bing Web Search needs an API key and endpoint wiring")

    def fetch(self, url: str) -> str:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
