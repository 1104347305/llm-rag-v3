"""向量语义检索（FAISS / pgvector + 本地暴力兜底）。

支持通过 settings.vector_store_type 切换后端:
  - "faiss": FAISS IndexFlatIP 本地 ANN 搜索
  - "pgvector": PostgreSQL pgvector HNSW 搜索

当向量存储不可用时，search_local() 提供本地暴力余弦搜索作为兜底方案。
"""
from __future__ import annotations

from src.main.python.config import settings
from src.main.python.steps.indexing.embedding_builder import EmbeddingBuilder, cosine
from src.main.python.models import Chunk
from src.main.python.steps.retrieval.base import BaseRetriever
from src.main.python.steps.stores.faiss_store import FaissStore, FaissUnavailable
from src.main.python.steps.stores.pgvector_store import PgVectorStore, PgVectorUnavailable
from loguru import logger


def _create_store():
    """根据 settings.vector_store_type 创建对应的向量存储实例。"""
    if settings.vector_store_type == "pgvector":
        return PgVectorStore.get()
    return FaissStore()


class VectorRetriever(BaseRetriever):
    """向量语义检索器。

    两路策略：
    - search(): 向量存储 ANN 检索（FAISS 或 pgvector）。
    - search_local(): 本地暴力余弦搜索（测试/兜底）。
    """

    def __init__(self, store: FaissStore | PgVectorStore | None = None) -> None:
        self._store = store

    def _get_store(self):
        if self._store is None:
            self._store = _create_store()
        return self._store

    @property
    def is_available(self) -> bool:
        return settings.vector_store_enabled

    async def search(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        query_vector = await EmbeddingBuilder.embed(query, settings.embedding_dim)
        if not query_vector:
            return []
        try:
            return await self._get_store().search_vectors(project_id, query_vector, top_k)
        except (FaissUnavailable, PgVectorUnavailable) as exc:
            logger.bind(event="vector.search_unavailable").warning(
                f"{settings.vector_store_type} search unavailable", error=str(exc))
            return []

    @staticmethod
    async def search_local(query: str, chunks: list[Chunk], top_k: int = 100) -> list[tuple[str, float]]:
        query_vector = await EmbeddingBuilder.embed(query, settings.embedding_dim)
        results = [(c.chunk_id, cosine(query_vector, c.vector)) for c in chunks if c.vector]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
