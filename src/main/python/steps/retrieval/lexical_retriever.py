"""中文词召回检索（SQLite FTS5 + 本地中文分词评分）。

两路检索策略：
1. FTS5: SQLite 内置全文搜索引擎，提供 BM25 排序的快速关键词检索
2. 本地分词评分: 对查询做中文分词后，多维度匹配 chunk 的标题/标题/内容
"""
from __future__ import annotations

import re

from src.main.python.config import settings
from src.main.python.models import Chunk
from src.main.python.steps.retrieval.base import BaseRetriever
from src.main.python.db.local_store import LocalStore
from src.main.python.utils.logging import get_logger

logger = get_logger(__name__)


class LexicalRetriever(BaseRetriever):
    """中文词召回检索器。

    FTS5 BM25 优先（需要 SQLite 索引），本地分词评分作为兜底。

    使用示例:
        retriever = LexicalRetriever()

        # FTS5 检索
        results = retriever.search("保险条款", "my-project", top_k=100)

        # 本地分词评分（无需索引）
        if retriever.is_local_available(chunks):
            results = LexicalRetriever.search_local("保险条款", chunks, top_k=100)
    """

    # ═══════════════════════════════════════════════════════
    # 分词常量
    # ═══════════════════════════════════════════════════════

    STOP_WORDS: set[str] = {
        "的", "是", "了", "什么", "在", "有", "和", "与",
        "对", "从", "吗", "呢",
        "the", "is", "a", "an", "what", "how", "are",
        "in", "on", "at", "to", "for", "of",
    }

    TRIM_PUNCT_RE: re.Pattern = re.compile(
        r"^[\s,，。！？、；：\"'（）()\-_/\\·~～…]+|[\s,，。！？、；：\"'（）()\-_/\\·~～…]+$"
    )

    # ═══════════════════════════════════════════════════════
    # 构造与状态
    # ═══════════════════════════════════════════════════════

    def __init__(self, store: LocalStore | None = None) -> None:
        """初始化词召回检索器。

        Args:
            store: 本地存储实例。为 None 时使用全局单例 LocalStore.get()。
        """
        self._store = store or LocalStore.get()

    @property
    def is_available(self) -> bool:
        """FTS5 是否可用。检查 settings.enable_local_lexical_retrieval 功能开关。"""
        return settings.enable_local_lexical_retrieval

    # ═══════════════════════════════════════════════════════
    # 检索方法
    # ═══════════════════════════════════════════════════════

    def search(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        """FTS5 BM25 检索。

        要求 project_id 对应的 SQLite 索引已构建（has_sqlite_index 返回 True）。
        无索引时返回空列表，调用方应回退到 search_local()。

        Args:
            query: 查询文本。
            project_id: 项目标识。
            top_k: 返回的最大 chunk 数。

        Returns:
            [(chunk_id, bm25_score), ...]。无索引时返回 []。
        """
        if not self._store.has_sqlite_index(project_id):
            return []
        return self._store.search_chunks_fts(project_id, query, top_k)

    def is_local_available(self, chunks: list[Chunk]) -> bool:
        """判断本地分词评分是否可用。

        chunk 数量在 local_lexical_max_chunks 阈值内时才启用，
        避免全量内存扫描导致延迟过高。

        Args:
            chunks: 候选 Chunk 列表。

        Returns:
            True 表示本地分词评分可以执行。
        """
        return len(chunks) <= settings.local_lexical_max_chunks

    @staticmethod
    def search_local(query: str, chunks: list[Chunk], top_k: int = 100) -> list[tuple[str, float]]:
        """本地中文词汇评分检索。

        不依赖任何外部索引，在内存中对所有 chunk 执行多维度分词匹配打分。

        Args:
            query: 查询文本。
            chunks: 所有候选 Chunk 列表。
            top_k: 返回的最大结果数。

        Returns:
            [(chunk_id, lexical_score), ...]，按分数降序。
        """
        return LexicalRetriever.lexical_search(query, chunks, top_k)

    # ═══════════════════════════════════════════════════════
    # 中文分词与评分算法
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def lexical_search(query: str, chunks: list[Chunk], top_k: int = 100) -> list[tuple[str, float]]:
        """遍历所有 chunk 执行分词匹配打分。

        流程：去除标点 → 中文分词 → 对每个 chunk 调用 score_chunk 多维度打分
        → 过滤零分项 → 按分数降序取 top_k。

        Args:
            query: 查询文本。
            chunks: 所有候选 Chunk。
            top_k: 返回的最大结果数。

        Returns:
            [(chunk_id, score), ...]，按分数降序。仅包含 score > 0 的结果。
        """
        # 去除首尾标点，得到纯查询短语
        query_phrase = LexicalRetriever.TRIM_PUNCT_RE.sub("", query.strip().lower())
        tokens = LexicalRetriever.tokenize_query(query)
        if not query_phrase and not tokens:
            return []

        results: list[tuple[str, float]] = []
        for chunk in chunks:
            score = LexicalRetriever.score_chunk(chunk, query_phrase, tokens)
            if score > 0:
                results.append((chunk.chunk_id, score))
        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    @staticmethod
    def tokenize_query(query: str) -> list[str]:
        """中文查询分词。

        策略：
        1. 按标点/空格切分，过滤停用词
        2. 对 CJK token（长度 > 2）生成：
           - 二元组（相邻双字，如 "家庭医生" → "家庭", "庭医", "医生"）
           - 去停用词后的单字
           - 完整 token
        3. 非 CJK token 原样保留
        4. 去重（保持顺序）

        Args:
            query: 已小写的查询文本。

        Returns:
            去重后的 token 列表。
        """
        # 步骤 1: 按标点切分，过滤停用词
        raw_tokens = [
            token
            for token in re.split(r"[\s,，。！？、；：\"'（）()\-_/\\·~～…]+", query.lower())
            if token and token not in LexicalRetriever.STOP_WORDS
        ]
        tokens: list[str] = []
        for token in raw_tokens:
            has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", token))
            if has_cjk and len(token) > 2:
                chars = list(token)
                # 二元组: "家庭医生" → ["家庭", "庭医", "医生"]
                tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
                # 单字（去停用词）
                tokens.extend(ch for ch in chars if ch not in LexicalRetriever.STOP_WORDS)
                # 完整词
                tokens.append(token)
            else:
                tokens.append(token)
        # 步骤 4: 去重保持顺序
        return list(dict.fromkeys(tokens))

    @staticmethod
    def score_chunk(chunk: Chunk, query_phrase: str, tokens: list[str]) -> float:
        """对单个 chunk 执行多维度评分。

        评分由两部分累加：

        A) 短语匹配（query_phrase）：
           - 精确标题匹配 → lexical_exact_title_score
           - 标题包含 → lexical_phrase_title_score
           - 标题包含 → lexical_phrase_heading_score
           - 内容出现次数（上限 lexical_phrase_content_limit）→ lexical_phrase_content_score

        B) Token 匹配（tokens）：
           - 标题命中 → lexical_token_title_score
           - 标题命中 → lexical_token_heading_score
           - 内容命中 → lexical_token_content_score

        最后乘以 chunk 类型 boost（entity/concept/source）。

        Args:
            chunk: 待评分的 Chunk 对象。
            query_phrase: 去除首尾标点后的查询短语。
            tokens: 分词后的 token 列表。

        Returns:
            累加后的评分（浮点数）。
        """
        title_text = f"{chunk.title} {chunk.path}".lower()
        heading_text = chunk.heading_path.lower()
        content_text = chunk.content.lower()
        score = 0.0

        # ── A) 短语匹配 ──
        if query_phrase:
            if chunk.title.lower() == query_phrase:
                score += settings.lexical_exact_title_score
            if query_phrase in title_text:
                score += settings.lexical_phrase_title_score
            if query_phrase in heading_text:
                score += settings.lexical_phrase_heading_score
            # 内容出现次数，有上限防止长文档过度加权
            score += min(content_text.count(query_phrase), settings.lexical_phrase_content_limit) * settings.lexical_phrase_content_score

        # ── B) Token 匹配 ──
        for token in tokens:
            if token in title_text:
                score += settings.lexical_token_title_score
            if token in heading_text:
                score += settings.lexical_token_heading_score
            if token in content_text:
                score += settings.lexical_token_content_score

        # ── 类型 boost ──
        _type_boost = {
            "entity": settings.lexical_chunk_type_entity_boost,
            "concept": settings.lexical_chunk_type_concept_boost,
            "source": settings.lexical_chunk_type_source_boost,
        }
        boost = _type_boost.get(chunk.type)
        if boost is not None:
            score *= boost

        return score
