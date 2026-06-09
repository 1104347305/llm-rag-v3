from __future__ import annotations

from src.main.python.models import Chunk
from src.main.python.db.dashscope import DashScopeUnavailable, get_dashscope_client
from src.main.python.utils.logging import get_logger, log_event
from src.main.python.utils.text import tokenize

logger = get_logger(__name__)


class Reranker:
    """重排序器：DashScope API 优先，不可用时本地 token 重叠打分。"""

    def __init__(self) -> None:
        """初始化重排序器，获取 DashScope 客户端。"""

        self._client = get_dashscope_client()

    def score_chunks(self, query: str, chunks: list[Chunk], top_n: int | None = None) -> dict[str, float]:
        """对 chunks 重排序，返回 {chunk_id: score}。"""
        if not chunks:
            return {}
        documents = [
            f"Title: {c.title}\nSection: {c.heading_path}\nContent: {c.content}"
            for c in chunks
        ]
        try:
            results = self._client.rerank(query, documents, top_n=top_n or len(documents))
        except DashScopeUnavailable as exc:
            log_event(logger, 30, "rerank.fallback_local",
                      "reranker unavailable; using local overlap scorer",
                      error=str(exc), document_count=len(documents))
            return {c.chunk_id: self._local_score(query, c) for c in chunks}

        scores: dict[str, float] = {}
        for item in results:
            if 0 <= item["index"] < len(chunks):
                scores[chunks[item["index"]].chunk_id] = item["score"]
        for c in chunks:
            scores.setdefault(c.chunk_id, 0.0)
        return scores

    @staticmethod
    def _local_score(query: str, chunk: Chunk) -> float:
        """本地 token 重叠打分（兜底方案）。"""
        query_terms = set(tokenize(query))
        if not query_terms:
            return 0.0
        title_overlap = len(query_terms & set(tokenize(chunk.title))) / len(query_terms)
        heading_overlap = len(query_terms & set(tokenize(chunk.heading_path))) / len(query_terms)
        content_overlap = len(query_terms & set(tokenize(chunk.content))) / len(query_terms)
        return min(1.0, 0.65 * content_overlap + 0.2 * title_overlap + 0.15 * heading_overlap)

    @staticmethod
    def rerank_score(query: str, chunk: Chunk) -> float:
        """单个 chunk 本地评分（不依赖 DashScope，用于图扩展等场景）。"""
        return Reranker._local_score(query, chunk)

    @staticmethod
    def rerank_chunks(query: str, chunks: list[Chunk], top_n: int | None = None) -> dict[str, float]:
        """批量重排序（DashScope 优先，不可用时本地打分）。"""
        return Reranker().score_chunks(query, chunks, top_n)
