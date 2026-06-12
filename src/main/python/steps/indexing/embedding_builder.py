"""文本嵌入构建器——DashScope API + 本地 hash 兜底。"""
from __future__ import annotations

import hashlib
import math
from tqdm import tqdm
from src.main.python.config import settings
from src.main.python.services.request_service import DashScopeUnavailable, get_dashscope_client
from loguru import logger
from src.main.python.utils.text import tokenize



class EmbeddingBuilder:
    """文本嵌入构建器。

    DashScope API 优先，不可用时回退到 MD5 hash 伪嵌入。
    支持单条嵌入、批量嵌入和余弦相似度计算。
    """

    @staticmethod
    async def embed(text: str, dim: int = 0) -> list[float]:
        """单条文本嵌入。

        Args:
            text: 待嵌入文本。
            dim: 向量维度，0 = 自动。

        Returns:
            归一化向量列表。
        """
        try:
            return await get_dashscope_client().embed(text)
        except DashScopeUnavailable as exc:
            fallback_dim = dim or settings.fallback_embedding_dim
            logger.bind(event="embedding.fallback_hash").warning(
                f"embedding unavailable; using hash | error={exc} | text_chars={len(text)} | dim={fallback_dim}")
            return EmbeddingBuilder._hash_embed(text, fallback_dim)

    @staticmethod
    async def batch_embed(texts: list[str], dim: int = 0, batch_size: int | None = None) -> list[list[float]]:
        """批量嵌入，减少 API 调用次数。

        Args:
            texts: 文本列表。
            dim: 向量维度。
            batch_size: 每批条数，默认取自 settings.embedding_batch_size。

        Returns:
            向量列表。
        """
        if batch_size is None:
            batch_size = settings.embedding_batch_size
        if not texts:
            return []
        client = get_dashscope_client()
        all_vectors: list[list[float]] = []
        for start in tqdm(range(0, len(texts), batch_size), desc='embeddings索引更新：'):
            batch = texts[start:start + batch_size]
            try:
                all_vectors.extend(await client.batch_embed(batch))
            except DashScopeUnavailable as exc:
                logger.bind(event="embedding.batch_fallback").warning(
                    f"batch embedding failed; using hash | error={exc} | batch_size={len(batch)}")
                for text in batch:
                    all_vectors.append(EmbeddingBuilder._hash_embed(text, dim or 384))
        return all_vectors

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        """余弦相似度（假设向量已归一化，直接点积）。"""
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> list[float]:
        """MD5 hash 伪嵌入兜底。"""
        vector = [0.0] * dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


# 向后兼容
cosine = EmbeddingBuilder.cosine
embed_text = EmbeddingBuilder.embed
batch_embed = EmbeddingBuilder.batch_embed
