from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI


@dataclass(slots=True)
class ChatUsage:
    """API 返回的 token 用量。"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def ratio(self, context_window: int) -> float:
        """prompt 占上下文窗口的比例。"""
        if context_window <= 0:
            return 0.0
        return self.prompt_tokens / context_window


@dataclass(slots=True)
class LLMClient:
    api_key: str
    base_url: str
    model: str
    _client: OpenAI = field(init=False)
    last_usage: ChatUsage | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        # 记录本次用量（副作用），供后续压缩判断使用
        if response.usage:
            self.last_usage = ChatUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return response.choices[0].message.content or ""

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        """多轮对话（用于 ReAct Loop）。

        Args:
            messages: 标准 OpenAI messages 列表
            temperature: 生成温度
            tools: Function calling 工具定义（可选）
            tool_choice: 工具选择策略（可选）

        Returns:
            dict with keys:
              - content: str (文本回复内容)
              - tool_calls: list[dict] | None (工具调用)
              - usage: ChatUsage | None
        """
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**kwargs)

        if response.usage:
            self.last_usage = ChatUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        msg = response.choices[0].message
        result: dict[str, Any] = {
            "content": msg.content or "",
            "tool_calls": None,
            "usage": self.last_usage,
        }
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        return result

    def embed(self, texts: list[str], embedding_model: str = "text-embedding-3-small") -> list[list[float]]:
        response = self._client.embeddings.create(model=embedding_model, input=texts)
        return [item.embedding for item in response.data]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算文本的 token 数。

        对英文为主的文本：约 4 字符/token
        对中文为主的文本：约 1-2 字符/token
        取保守估计（偏大），避免压缩触发过晚。
        """
        if not text:
            return 0
        chars = len(text)
        cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        non_cjk = chars - cjk_count
        # CJK: ~1.5 chars/token; 非 CJK: ~4 chars/token; 取保守值
        return int(cjk_count * 1.5 + non_cjk / 3.5) + 8  # +8 为消息框架开销


# 模块级模型缓存 —— 整个进程生命周期内只加载一次 SentenceTransformer
_MODEL_CACHE: dict[str, tuple[int, object]] = {}

# 模型本地存储目录（相对于 AURADERMA_DATA_DIR）
_MODEL_CACHE_REL_DIR = "models"


class LocalEmbedder:
    """本地 embedding 模型（中文优化），用于向量检索。

    模型只在首次调用时从 HuggingFace 下载一次，保存在 ``data/models/`` 目录下，
    后续启动直接加载本地缓存，不再联网。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model_name = model_name

        # 如果已缓存，直接复用模型对象，跳过全部加载
        if model_name in _MODEL_CACHE:
            self.dim, self._model = _MODEL_CACHE[model_name]
            return

        from sentence_transformers import SentenceTransformer

        # 使用项目内的本地缓存目录，而非 HuggingFace 默认的 ~/.cache/huggingface/
        cache_dir = self._resolve_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [model] 加载 embedding 模型 {model_name} ...", flush=True)
        self._model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        self.dim = self._model.get_embedding_dimension()
        print(f"  [model] 加载完成，向量维度={self.dim}", flush=True)

        # 写入模块级缓存
        _MODEL_CACHE[model_name] = (self.dim, self._model)

    @staticmethod
    def _resolve_cache_dir() -> Path:
        """解析模型缓存目录路径。"""
        import os

        data_dir = os.getenv("AURADERMA_DATA_DIR", "./data")
        return Path(data_dir).resolve() / _MODEL_CACHE_REL_DIR

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
