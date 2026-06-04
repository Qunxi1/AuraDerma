from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(slots=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(self, client: QdrantClient, products_collection: str, memory_collection: str, docs_collection: str) -> None:
        self.client = client
        self.products_collection = products_collection
        self.memory_collection = memory_collection
        self.docs_collection = docs_collection

    def search(self, collection: str, query_vector: list[float], limit: int = 5, filters: models.Filter | None = None) -> list[SearchHit]:
        # query_points 兼容本地模式和服务器模式
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=filters,
            with_payload=True,
        )
        return [SearchHit(id=str(hit.id), score=hit.score or 0.0, payload=hit.payload or {}) for hit in results.points]

    def upsert_payload(self, collection: str, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def ensure_collections(self, vector_size: int = 1536) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        for name in [self.products_collection, self.memory_collection, self.docs_collection]:
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
            else:
                # 校验已有 collection 的向量维度是否匹配
                info = self.client.get_collection(name)
                existing_size = info.config.params.vectors.size
                if existing_size != vector_size:
                    raise ValueError(
                        f"Collection '{name}' 向量维度为 {existing_size}，"
                        f"但当前模型要求 {vector_size}。请删除旧 collection 后重试。"
                    )
