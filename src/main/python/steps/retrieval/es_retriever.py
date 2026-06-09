"""ES BM25 词汇精确匹配检索引擎。

Elasticsearch 实现 BM25 算法，擅长关键词精确匹配和词汇级召回。
"""
from __future__ import annotations

from src.main.python.config import settings
from src.main.python.steps.retrieval.base import BaseRetriever
from src.main.python.db.elasticsearch import ElasticsearchClient, ElasticsearchUnavailable
from src.main.python.utils.logging import get_logger

logger = get_logger(__name__)


class EsRetriever(BaseRetriever):
    """ES BM25 词汇精确匹配检索引擎。

    依赖注入 ES 客户端，未提供时自动获取全局单例。

    使用示例:
        retriever = EsRetriever()                     # 使用默认 ES 连接
        retriever = EsRetriever(client=custom_client)  # 注入自定义客户端
        results = retriever.search("保险条款", "my-project", top_k=100)
    """

    def __init__(self, client: ElasticsearchClient | None = None) -> None:
        """初始化 ES 检索器。client 延迟获取，首次 search() 时才连接。

        Args:
            client: ES 客户端。为 None 时首次调用自动获取全局单例。
        """
        self._client = client

    @property
    def is_available(self) -> bool:
        """ES 是否配置且可用。检查 settings.es_retrieval_enabled（URL + 功能开关）。"""
        return settings.es_retrieval_enabled

    def _get_client(self) -> ElasticsearchClient:
        """延迟获取 ES 客户端单例。"""
        if self._client is None:
            self._client = ElasticsearchClient.get()
        return self._client

    def search(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        """BM25 词汇检索。

        Args:
            query: 查询文本。
            project_id: 项目标识。
            top_k: 返回的最大 chunk 数，默认 100。

        Returns:
            [(chunk_id, bm25_score), ...]，按分数降序。

        Raises:
            ElasticsearchUnavailable: ES 不可用时抛出，调用方应捕获并降级。
        """
        return self._get_client().search_chunks(project_id, query, top_k)
