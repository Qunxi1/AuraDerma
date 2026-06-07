from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI


@dataclass(slots=True)
class LLMClient:
    api_key: str
    base_url: str
    model: str
    _client: OpenAI = field(init=False)

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
        return response.choices[0].message.content or ""

    def embed(self, texts: list[str], embedding_model: str = "text-embedding-3-small") -> list[list[float]]:
        response = self._client.embeddings.create(model=embedding_model, input=texts)
        return [item.embedding for item in response.data]


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
