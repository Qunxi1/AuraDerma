from __future__ import annotations

from dataclasses import dataclass, field
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


class LocalEmbedder:
    """本地 embedding 模型（中文优化），用于向量检索。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
