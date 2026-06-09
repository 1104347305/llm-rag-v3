"""向量语义检索（pgvector + 本地暴力兜底）。

pgvector 提供高效的 ANN 向量搜索。当 pgvector 不可用时，
search_local() 提供本地暴力余弦搜索作为兜底方案。
"""
from __future__ import annotations

from src.main.python.config import settings
from src.main.python.steps.indexing.embedding_builder import cosine, embed_text
from src.main.python.models import Chunk
from src.main.python.steps.retrieval.base import BaseRetriever
from src.main.python.db.pgvector_store import PgVectorStore, PgVectorUnavailable
from src.main.python.utils.logging import get_logger, log_event

logger = get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """向量语义检索器。

    两路策略：
    - search(): pgvector ANN 索引检索（生产环境）。
    - search_local(): 本地暴力余弦搜索（测试/兜底）。

    使用示例:
        retriever = VectorRetriever()
        results = retriever.search("保险条款", "my-project", top_k=50)

        # 本地暴力搜索（无需 pgvector）：
        results = VectorRetriever.search_local("保险条款", chunks, top_k=50)
    """

    def __init__(self, store: PgVectorStore | None = None) -> None:
        """初始化向量检索器。store 延迟获取，首次 search() 时才连接。

        Args:
            store: PgVector 存储实例。为 None 时首次调用自动获取全局单例。
        """
        self._store = store

    def _get_store(self) -> PgVectorStore:
        """延迟获取 pgvector 存储单例。"""
        if self._store is None:
            self._store = PgVectorStore.get()
        return self._store

    @property
    def is_available(self) -> bool:
        """pgvector 是否配置且可用。检查 settings.pgvector_enabled（主机地址 + 功能开关）。"""
        return settings.pgvector_enabled

    def search(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        """pgvector 语义搜索。

        流程：embed_text 生成查询向量 → pgvector ANN 搜索 → 返回 top_k。

        Args:
            query: 查询文本。
            project_id: 项目标识。
            top_k: 返回的最大 chunk 数。

        Returns:
            [(chunk_id, cosine_similarity), ...]。pgvector 不可用时返回空列表。
        """
        query_vector = embed_text(query, settings.embedding_dim)
        if not query_vector:
            return []
        try:
            return self._get_store().search_vectors(project_id, query_vector, top_k)
        except PgVectorUnavailable as exc:
            log_event(logger, 30, "vector.pgvector_unavailable",
                      "pgvector search unavailable", error=str(exc))
            return []

    @staticmethod
    def search_local(query: str, chunks: list[Chunk], top_k: int = 100) -> list[tuple[str, float]]:
        """本地暴力余弦搜索（兜底方案）。

        不依赖 pgvector，直接在内存中对所有 chunk 计算余弦相似度。
        适用于 chunk 数量较小（< local_vector_max_chunks）的场景。

        Args:
            query: 查询文本。
            chunks: 所有候选 Chunk 列表。
            top_k: 返回的最大结果数。

        Returns:
            [(chunk_id, cosine_similarity), ...]，按相似度降序。
        """
        query_vector = embed_text(query, settings.embedding_dim)
        # 对每个有向量的 chunk 计算余弦相似度
        results = [(c.chunk_id, cosine(query_vector, c.vector)) for c in chunks if c.vector]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
