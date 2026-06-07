from __future__ import annotations

from skills.web_search.client import WebSearchClient


class WebSearchSkill:
    """Adapter that wraps WebSearchClient into a tool-payload interface."""

    def __init__(self, client: WebSearchClient) -> None:
        self._client = client

    def to_tool_payload(self, query: str, top_k: int = 5) -> dict:
        results = self._client.search(query, top_k=top_k)
        return {
            "name": "web_search",
            "description": "Search the public web for current information",
            "query": query,
            "top_k": top_k,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in results
            ],
        }
