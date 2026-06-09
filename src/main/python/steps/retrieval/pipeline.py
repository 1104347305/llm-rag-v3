# ============================================================
# 混合检索 Pipeline（主入口）
# ============================================================
# 三路并行召回 → RRF 融合 → 重排序 → 图扩展 → 上下文构建:
#
#   路 1: ES BM25（词汇精确匹配）      ← enable_es_retrieval
#   路 2: pgvector（语义相似度）        ← enable_vector_retrieval
#   路 3: SQLite FTS5（本地全文搜索）   ← enable_local_lexical_retrieval
#          ↓
#   RRF 融合（Reciprocal Rank Fusion, k=60）
#          ↓
#   候选 SQLite 加载（按 chunk_id 取全文）
#          ↓
#   重排序（DashScope / 本地 token 重叠）
#          ↓
#   图扩展 + 相邻块扩展
#          ↓
#   上下文构建 → 返回
#
# 两条路径：
#   - 候选路径：三路召回有结果 + SQLite 索引可用 → SQLite 按需加载候选
#   - 兜底路径：无候选或索引不可用 → 全量加载内存 → 词召回 + 暴力向量
#
# 候选评分权重:
#   0.25 page_rrf + 0.35 chunk_rrf + 0.20 rerank
#   + 0.10 title_match + 0.10 graph + 0.05 metadata
# ============================================================

from __future__ import annotations

import re
import threading

from src.main.python.config import settings
from src.main.python.models import Chunk, Page
from src.main.python.steps.retrieval.context_builder import ContextBuilder
from src.main.python.steps.retrieval.es_retriever import EsRetriever
from src.main.python.steps.retrieval.graph_expand import GraphExpander
from src.main.python.steps.retrieval.lexical_retriever import LexicalRetriever
from src.main.python.steps.retrieval.page_aggregation import aggregate_pages
from src.main.python.steps.retrieval.reranker import Reranker
from src.main.python.steps.retrieval.vector_retriever import VectorRetriever
from src.main.python.db.elasticsearch import ElasticsearchUnavailable
from src.main.python.db.local_store import LocalStore
from src.main.python.utils.logging import get_logger, log_event
from src.main.python.utils.timers import timer, trace

logger = get_logger(__name__)


class RetrievalPipeline:
    """混合检索 Pipeline，通过依赖注入组装各模块。

    使用示例:
        pipeline = RetrievalPipeline.get()                       # 使用全局单例
        pipeline = RetrievalPipeline(es=custom_es, vector=...)    # 注入自定义模块
        result = pipeline.retrieve("my-project", "保险查询", top_pages=5)
    """

    # ── 常量 ────────────────────────────────────────────────

    TRIM_PUNCT_RE: re.Pattern = re.compile(
        r"^[\s,，。！？、；：\"'（）()\-_/\\·~～…]+|[\s,，。！？、；：\"'（）()\-_/\\·~～…]+$"
    )
    """标点修剪正则，用于 _title_match_boost 查询/标题归一化。"""

    _instance: RetrievalPipeline | None = None
    """全局单例，由 get() 延迟初始化。"""
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> RetrievalPipeline:
        """获取全局单例（线程安全），首次调用时初始化。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self,
                 store: LocalStore | None = None,
                 es: EsRetriever | None = None,
                 vector: VectorRetriever | None = None,
                 lexical: LexicalRetriever | None = None,
                 reranker: Reranker | None = None,
                 graph_expander: GraphExpander | None = None,
                 ctx_builder: ContextBuilder | None = None) -> None:
        """依赖注入构造函数。所有参数可选，为 None 时自动使用全局单例。

        Args:
            store: 本地存储（SQLite 索引 + 全量加载）。
            es: ES BM25 检索器。
            vector: 向量检索器（pgvector + 本地暴力）。
            lexical: 中文词召回检索器（FTS5 + 分词评分）。
            reranker: 重排序器（DashScope / 本地 token 重叠）。
            graph_expander: 图扩展器。
            ctx_builder: 上下文构建器。
        """
        self._store = store or LocalStore.get()
        self._es = es or EsRetriever()
        self._vector = vector or VectorRetriever()
        self._lexical = lexical or LexicalRetriever()
        self._reranker = reranker or Reranker()
        self._graph_expander = graph_expander or GraphExpander()
        self._ctx_builder = ctx_builder or ContextBuilder()

    # ── 检索代理（可供测试 mock）───────────────────────────

    def _bm25_search(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        """ES BM25 检索代理。"""
        return self._es.search(query, project_id, top_k)

    def _vector_search_pgvector(self, query: str, project_id: str, top_k: int = 100) -> list[tuple[str, float]]:
        """pgvector 语义检索代理。"""
        return self._vector.search(query, project_id, top_k)

    @staticmethod
    def _vector_search(query: str, chunks: list[Chunk], top_k: int = 100) -> list[tuple[str, float]]:
        """本地暴力向量检索代理。"""
        return VectorRetriever.search_local(query, chunks, top_k)

    # ── 公共入口 ────────────────────────────────────────────

    def retrieve(self,
                 project_id: str, query: str,
                 max_context_size: int | None = None,
                 top_pages: int | None = None,
                 bm25_top_k: int | None = None,
                 vector_top_k: int | None = None,
                 rerank_top_k: int | None = None,
                 include_es: bool | None = None,
                 include_vector: bool | None = None,
                 include_lexical: bool | None = None,
                 include_graph: bool = True,
                 include_neighbor_chunks: bool = True,
                 debug: bool = False,
                 ) -> dict[str, object]:
        """执行混合检索，返回 LLM 可用的上下文。

        Args:
            project_id: 项目标识。
            query: 用户查询文本。
            max_context_size: LLM 上下文最大字符数，默认 settings.default_max_context_size。
            top_pages: 最终返回的最大页面数，默认 settings.default_top_pages。
            bm25_top_k: ES/FTS5 召回的最大 chunk 数，默认 settings.default_bm25_top_k。
            vector_top_k: pgvector 召回的最大 chunk 数，默认 settings.default_vector_top_k。
            rerank_top_k: 送入重排序的候选 chunk 数，默认 settings.default_rerank_top_k。
            include_es: 是否启用 ES 路。None = 使用全局配置。
            include_vector: 是否启用向量路。None = 使用全局配置。
            include_lexical: 是否启用 FTS5 路。None = 使用全局配置。
            include_graph: 是否启用图扩展。
            include_neighbor_chunks: 是否启用相邻块扩展。
            debug: 是否在返回中包含 debug 详情。

        Returns:
            dict，包含 pages, page_list, pages_context, index, metrics 等字段。
        """
        # ── 参数默认值解析 ──
        max_context_size = max_context_size or settings.default_max_context_size
        top_pages = top_pages or settings.default_top_pages
        bm25_top_k = bm25_top_k or settings.default_bm25_top_k
        vector_top_k = vector_top_k or settings.default_vector_top_k
        rerank_top_k = rerank_top_k or settings.default_rerank_top_k
        metrics: dict[str, float] = {}
        reasons: list[str] = []

        # 各路开关：None = 使用全局配置，显式 True/False 覆盖
        use_es = settings.es_retrieval_enabled if include_es is None else include_es
        use_vector = settings.enable_vector_retrieval if include_vector is None else include_vector
        use_lexical = settings.enable_local_lexical_retrieval if include_lexical is None else include_lexical

        log_event(logger, 20, "retrieval.start", "retrieve context started",
                  project_id=project_id, query_length=len(query),
                  max_context_size=max_context_size, top_pages=top_pages,
                  bm25_top_k=bm25_top_k, vector_top_k=vector_top_k, rerank_top_k=rerank_top_k,
                  use_es=use_es, use_vector=use_vector, use_lexical=use_lexical,
                  include_graph=include_graph, include_neighbor_chunks=include_neighbor_chunks)

        # ── 路 1: ES BM25（词汇精确匹配）─────────────────────────
        with trace(metrics, "bm25_latency_ms", logger, "ES BM25 词汇检索",
                   input=dict(query=query, top_k=bm25_top_k)) as out:
            bm25_results: list[tuple[str, float]] = []
            if use_es:
                try:
                    bm25_results = self._bm25_search(query, project_id, bm25_top_k)
                    out["count"] = len(bm25_results)
                    out["top_scores"] = [s for _, s in bm25_results[:3]]
                except ElasticsearchUnavailable as exc:
                    out["error"] = str(exc)
            else:
                out["status"] = "disabled"

        # ── 路 2: pgvector（语义相似度）──────────────────────────
        with trace(metrics, "vector_latency_ms", logger, "pgvector 语义检索",
                   input=dict(query=query, top_k=vector_top_k)) as out:
            vector_results: list[tuple[str, float]] = []
            if use_vector:
                vector_results = self._vector_search_pgvector(query, project_id, vector_top_k)
                out["count"] = len(vector_results)
                out["top_scores"] = [round(s, 4) for _, s in vector_results[:3]]
            else:
                out["status"] = "disabled"

        # ── 路 3: SQLite FTS5（本地全文搜索）─────────────────────
        with trace(metrics, "fts5_latency_ms", logger, "FTS5 本地全文搜索",
                   input=dict(query=query, top_k=bm25_top_k)) as out:
            fts5_results: list[tuple[str, float]] = []
            if use_lexical and self._store.has_sqlite_index(project_id):
                fts5_results = self._lexical.search(query, project_id, bm25_top_k)
                out["count"] = len(fts5_results)
                out["top_scores"] = [round(s, 2) for _, s in fts5_results[:3]]
            else:
                out["status"] = "disabled" if not use_lexical else "no_index"

        # ── 路由决策：候选路径 vs 兜底路径 ──
        has_candidates = bool(bm25_results or vector_results or fts5_results)
        if has_candidates and self._store.has_sqlite_index(project_id):
            log_event(logger, 20, "retrieval.mode", "hybrid candidate retrieval",
                      project_id=project_id, mode="candidate_sqlite",
                      bm25_count=len(bm25_results), vector_count=len(vector_results),
                      fts5_count=len(fts5_results), reasons=reasons)
            return self._retrieve_from_candidates(
                project_id=project_id, query=query,
                bm25_results=bm25_results, vector_results=vector_results,
                fts5_results=fts5_results,
                max_context_size=max_context_size, top_pages=top_pages,
                rerank_top_k=rerank_top_k, include_graph=include_graph,
                include_neighbor_chunks=include_neighbor_chunks,
                debug=debug, metrics=metrics, reasons=reasons)

        # ── 兜底路径：全量加载 + 词召回 + 暴力向量 ─────────────
        return self._full_local_fallback(
            project_id=project_id, query=query,
            max_context_size=max_context_size, top_pages=top_pages,
            rerank_top_k=rerank_top_k, include_graph=include_graph,
            include_neighbor_chunks=include_neighbor_chunks,
            use_vector=use_vector, bm25_top_k=bm25_top_k,
            vector_top_k=vector_top_k,
            debug=debug, metrics=metrics, reasons=reasons)

    # ── 候选路径 ────────────────────────────────────────────

    def _retrieve_from_candidates(
        self, project_id: str, query: str,
        bm25_results: list[tuple[str, float]],
        vector_results: list[tuple[str, float]],
        fts5_results: list[tuple[str, float]],
        max_context_size: int, top_pages: int, rerank_top_k: int,
        include_graph: bool, include_neighbor_chunks: bool,
        debug: bool, metrics: dict[str, float], reasons: list[str],
    ) -> dict[str, object]:
        """候选路径：三路召回 → RRF 融合 → SQLite 按需加载 → 评分 → 上下文。"""
        # 步骤 1: 从 SQLite 按 chunk_id 加载候选
        with trace(metrics, "candidate_load_latency_ms", logger, "候选数据加载",
                   input=dict(candidate_ids=len(bm25_results)+len(vector_results)+len(fts5_results))) as out:
            all_results = bm25_results + vector_results + fts5_results
            chunk_ids = [cid for cid, _ in all_results[:max(rerank_top_k, top_pages * settings.candidate_load_factor)]]
            chunks = self._store.load_chunks_by_ids(project_id, chunk_ids)
            chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            pages = self._store.load_pages_by_ids(project_id, [chunk.page_id for chunk in chunks])
            pages_by_id = {page.page_id: page for page in pages}
            out["chunks"] = len(chunks_by_id)
            out["pages"] = len(pages_by_id)

        # 步骤 2: RRF 融合
        with trace(metrics, "rrf_latency_ms", logger, "RRF 多路融合",
                   input=dict(bm25=len(bm25_results), vector=len(vector_results), fts5=len(fts5_results))) as out:
            fused = RetrievalPipeline.rrf_fuse({"bm25": bm25_results, "vector": vector_results, "fts5": fts5_results})
            out["fused"] = len(fused)

        # 步骤 3: 页面聚合（chunk → page 分组打分）
        with trace(metrics, "page_aggregation_latency_ms", logger, "页面聚合",
                   input=dict(fused_chunks=len(fused))) as out:
            page_results = aggregate_pages(fused, chunks_by_id, pages_by_id)
            out["pages"] = len(page_results)

        # 步骤 4: 图扩展
        graph_expansions: list[dict[str, object]] = []
        if include_graph:
            with trace(metrics, "graph_latency_ms", logger, "图扩展",
                       input=dict(candidate_pages=min(settings.graph_expand_limit_pages, len(page_results)))) as out:
                edges = self._store.load_edges_for_pages(
                    project_id,
                    [str(item["page_id"]) for item in page_results[:settings.graph_expand_limit_pages]],
                )
                graph_expansions = self._graph_expander.expand(page_results, edges)
                for page in self._store.load_pages_by_ids(project_id, [str(e["page_id"]) for e in graph_expansions if str(e["page_id"]) not in pages_by_id]):
                    pages_by_id[page.page_id] = page
                for chunk in self._store.load_chunks_for_pages(project_id, [str(e["page_id"]) for e in graph_expansions]):
                    chunks_by_id[chunk.chunk_id] = chunk
                out["edges"] = len(edges)
                out["expanded_pages"] = len(graph_expansions)

        # 步骤 5: 候选评分与选择
        with trace(metrics, "scoring_latency_ms", logger, "候选评分与选择",
                   input=dict(candidates=len(fused[:rerank_top_k]), top_pages=top_pages)) as out:
            candidate_chunk_ids = [str(item["id"]) for item in fused[:rerank_top_k]]
            for expansion in graph_expansions:
                candidate_chunk_ids.extend(
                    RetrievalPipeline._best_chunks_for_page(str(expansion["page_id"]), query, list(chunks_by_id.values())))
            selected = RetrievalPipeline._score_candidates(
                query, candidate_chunk_ids, chunks_by_id, page_results, graph_expansions, fused)[:top_pages]
            if not selected:
                selected = RetrievalPipeline._overview_fallback(list(pages_by_id.values()), list(chunks_by_id.values()))
            out["selected_pages"] = len(selected)
            out["titles"] = [
                pages_by_id.get(str(s["page_id"])).title if str(s["page_id"]) in pages_by_id else "?"
                for s in selected[:3]]

        # 步骤 6: 相邻块扩展
        if include_neighbor_chunks:
            self._load_neighbor_candidates(project_id, selected, chunks_by_id)
            for item in selected:
                page = pages_by_id[str(item["page_id"])]
                if page.type == "source":
                    item["chunk_ids"] = item["chunk_ids"][:settings.source_chunk_limit]
                else:
                    item["chunk_ids"] = RetrievalPipeline.expand_neighbor_chunks(item["chunk_ids"], chunks_by_id)

        # 步骤 7: 上下文构建
        with trace(metrics, "context_build_latency_ms", logger, "上下文构建",
                   input=dict(max_size=max_context_size)) as out:
            response = self._ctx_builder.build(selected, chunks_by_id, pages_by_id, max_context_size)
            pages_context = response.get("pages_context", "")
            out["context_chars"] = len(str(pages_context)) if pages_context else 0

        total_ms = sum(v for k, v in metrics.items() if k.endswith("_latency_ms"))
        logger.bind(event="trace").info(
            f"{total_ms:.0f}ms | 检索完成 | out: pages={len(selected)} mode=hybrid "
            f"context_chars={len(str(response.get('pages_context', '')))}")

        response["metrics"] = metrics
        response["retrieval_mode"] = "hybrid"
        if reasons:
            response["fallback_reasons"] = reasons
        if debug:
            response["debug"] = {
                "bm25": bm25_results[:20], "vector": vector_results[:20],
                "fts5": fts5_results[:20], "rrf": fused[:20],
                "pages": page_results[:20],
                "graph_expansions": graph_expansions, "selected": selected,
            }
        return response

    # ── 兜底路径 ────────────────────────────────────────────

    def _full_local_fallback(
        self, project_id: str, query: str,
        max_context_size: int, top_pages: int, rerank_top_k: int,
        include_graph: bool, include_neighbor_chunks: bool,
        use_vector: bool, bm25_top_k: int, vector_top_k: int,
        debug: bool, metrics: dict[str, float], reasons: list[str],
    ) -> dict[str, object]:
        """兜底路径：全量内存加载 → 词召回/暴力向量 → RRF → 评分 → 上下文。"""
        pages, chunks, edges = self._store.load_index(project_id)
        log_event(logger, 20, "retrieval.mode", "full local index retrieval",
                  project_id=project_id, mode="full_local",
                  page_count=len(pages), chunk_count=len(chunks), edge_count=len(edges))

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        pages_by_id = {page.page_id: page for page in pages}

        # 本地词召回（chunk 数量在阈值内时才执行）
        with timer(metrics, "lexical_latency_ms"):
            lexical_results: list[tuple[str, float]] = []
            if chunks and len(chunks) <= settings.local_lexical_max_chunks:
                lexical_results = LexicalRetriever.search_local(query, chunks, bm25_top_k)
            reasons.append(f"lexical_fallback: {len(lexical_results)} results")

        # 本地暴力向量搜索
        with timer(metrics, "vector_bruteforce_latency_ms"):
            full_vector_results: list[tuple[str, float]] = []
            if use_vector and len(chunks) <= settings.local_vector_max_chunks:
                full_vector_results = RetrievalPipeline._vector_search(query, chunks, vector_top_k)

        # RRF 融合
        with timer(metrics, "rrf_latency_ms"):
            fused = RetrievalPipeline.rrf_fuse(
                {"bm25": [], "lexical": lexical_results, "vector": full_vector_results})

        # 页面聚合
        with timer(metrics, "page_aggregation_latency_ms"):
            page_results = aggregate_pages(fused, chunks_by_id, pages_by_id)

        # 图扩展
        graph_expansions: list[dict[str, object]] = []
        if include_graph:
            with timer(metrics, "graph_latency_ms"):
                graph_expansions = self._graph_expander.expand(page_results, edges)

        # 候选评分
        chunks_list = list(chunks_by_id.values())
        candidate_chunk_ids = [str(item["id"]) for item in fused[:rerank_top_k]]
        for expansion in graph_expansions:
            candidate_chunk_ids.extend(
                RetrievalPipeline._best_chunks_for_page(str(expansion["page_id"]), query, chunks_list))
        selected = RetrievalPipeline._score_candidates(
            query, candidate_chunk_ids, chunks_by_id, page_results, graph_expansions, fused)[:top_pages]
        if not selected:
            selected = RetrievalPipeline._overview_fallback(pages, chunks_list)

        # 相邻块扩展
        if include_neighbor_chunks:
            for item in selected:
                page = pages_by_id[str(item["page_id"])]
                if page.type == "source":
                    item["chunk_ids"] = item["chunk_ids"][:settings.source_chunk_limit]
                else:
                    item["chunk_ids"] = RetrievalPipeline.expand_neighbor_chunks(item["chunk_ids"], chunks_by_id)

        # 上下文构建
        with timer(metrics, "context_build_latency_ms"):
            response = self._ctx_builder.build(selected, chunks_by_id, pages_by_id, max_context_size)

        response["metrics"] = metrics
        response["retrieval_mode"] = "full_local"
        if reasons:
            response["fallback_reasons"] = reasons
        if debug:
            response["debug"] = {
                "bm25": [], "lexical": lexical_results[:20], "vector": full_vector_results[:20],
                "rrf": fused[:20], "pages": page_results[:20],
                "graph_expansions": graph_expansions, "selected": selected,
            }
        return response

    # ── 辅助方法 ────────────────────────────────────────────

    def _load_neighbor_candidates(self, project_id: str,
                                   selected: list[dict[str, object]],
                                   chunks_by_id: dict[str, Chunk]) -> None:
        """从 SQLite 加载选中页面相邻块的 Chunk 对象到查找表中。"""
        neighbor_ids: list[str] = []
        for item in selected:
            for chunk_id in item.get("chunk_ids", []):
                chunk = chunks_by_id.get(str(chunk_id))
                if not chunk:
                    continue
                if chunk.prev_chunk_id:
                    neighbor_ids.append(chunk.prev_chunk_id)
                if chunk.next_chunk_id:
                    neighbor_ids.append(chunk.next_chunk_id)
        missing = [cid for cid in dict.fromkeys(neighbor_ids) if cid not in chunks_by_id]
        for chunk in self._store.load_chunks_by_ids(project_id, missing):
            chunks_by_id[chunk.chunk_id] = chunk

    # ── 静态辅助方法 ────────────────────────────────────────

    @staticmethod
    def _best_chunks_for_page(page_id: str, query: str, chunks: list[Chunk]) -> list[str]:
        """找到指定页面中 rerank_score 最高的 1 个 chunk_id（图扩展的页面代表）。"""
        page_chunks = [chunk for chunk in chunks if chunk.page_id == page_id]
        ranked = sorted(page_chunks, key=lambda chunk: Reranker.rerank_score(query, chunk), reverse=True)
        return [chunk.chunk_id for chunk in ranked[:1]]

    @staticmethod
    def _score_candidates(
        query: str, candidate_chunk_ids: list[str], chunks_by_id: dict[str, Chunk],
        page_results: list[dict[str, object]], graph_expansions: list[dict[str, object]],
        fused_chunks: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """多因子候选评分：加权总分 → 按页面分组 → 每页取 top scored_chunks_per_page。

        评分公式（权重来自 settings）:
            final = 0.25*page_rrf/max_rrf + 0.35*chunk_rrf/max_chunk_rrf
                  + 0.20*rerank + 0.10*title_match + 0.10*graph/4.0 + 0.05*type
        """
        page_rrf = {str(item["page_id"]): float(item["score"]) for item in page_results}
        graph_scores = {str(item["page_id"]): float(item["graph_score"]) for item in graph_expansions}
        max_rrf = max(page_rrf.values(), default=1.0)

        chunk_rrf = {str(item["id"]): float(item["score"]) for item in fused_chunks}
        max_chunk_rrf = max(chunk_rrf.values(), default=1.0)

        ordered_candidate_chunks = [chunks_by_id[cid] for cid in dict.fromkeys(candidate_chunk_ids) if cid in chunks_by_id]
        rerank_scores = Reranker.rerank_chunks(query, ordered_candidate_chunks)

        grouped: dict[str, dict[str, object]] = {}
        for chunk_id in dict.fromkeys(candidate_chunk_ids):
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            final_score = (
                settings.score_weight_page_rrf * page_rrf.get(chunk.page_id, 0.0) / max_rrf
                + settings.score_weight_chunk_rrf * chunk_rrf.get(chunk.chunk_id, 0.0) / max_chunk_rrf
                + settings.score_weight_rerank * rerank_scores.get(chunk.chunk_id, 0.0)
                + settings.score_weight_title_match * RetrievalPipeline._title_match_boost(query, chunk.title)
                + settings.score_weight_graph * min(1.0, graph_scores.get(chunk.page_id, 0.0) / 4.0)
                + settings.score_weight_type * (1.0 if chunk.type in {"entity", "concept", "source"} else 0.5)
            )
            entry = grouped.setdefault(chunk.page_id, {
                "page_id": chunk.page_id, "score": 0.0, "scored_chunks": [],
                "source": "graph" if graph_scores.get(chunk.page_id, 0) else "hybrid",
            })
            entry["score"] = max(float(entry["score"]), final_score)
            entry["scored_chunks"].append((chunk.chunk_id, final_score))

        selected = []
        for entry in grouped.values():
            scored_chunks = sorted(entry.pop("scored_chunks"), key=lambda item: item[1], reverse=True)
            entry["chunk_ids"] = [chunk_id for chunk_id, _ in scored_chunks[:settings.scored_chunks_per_page]]
            selected.append(entry)
        return sorted(selected, key=lambda item: float(item["score"]), reverse=True)

    @staticmethod
    def _title_match_boost(query: str, title: str) -> float:
        """查询与页面标题匹配度奖励：精确=1.0，包含=0.75，无=0.0。"""
        query_phrase = RetrievalPipeline.TRIM_PUNCT_RE.sub("", query.strip().lower())
        title_lower = title.strip().lower()
        if not query_phrase or not title_lower:
            return 0.0
        if query_phrase == title_lower:
            return 1.0
        if query_phrase in title_lower or title_lower in query_phrase:
            return 0.75
        return 0.0

    @staticmethod
    def _overview_fallback(pages: list[Page], chunks: list[Chunk]) -> list[dict[str, object]]:
        """无页面选中时回退到 overview.md 的前 3 个 chunk。"""
        overview = next((page for page in pages
                         if page.path.endswith("overview.md") or page.title.lower() == "overview"), None)
        if not overview:
            return []
        chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.page_id == overview.page_id][:3]
        return [{"page_id": overview.page_id, "score": 0.0, "chunk_ids": chunk_ids, "source": "overview"}]

    @staticmethod
    def rrf_fuse(rankings: dict[str, list[tuple[str, float]]],
                 k: float | None = None) -> list[dict[str, object]]:
        """RRF (Reciprocal Rank Fusion) 多路结果融合。

        公式: score = Σ 1/(k + rank)，k 默认 60（来自 settings.rrf_k）。

        Args:
            rankings: 各路结果 {"bm25": [...], "vector": [...], ...}。
            k: RRF 平滑常数。

        Returns:
            [{"id", "score", "sources", "raw_scores"}, ...]，按融合分数降序。
        """
        k = k if k is not None else settings.rrf_k
        fused: dict[str, dict[str, object]] = {}
        for source, results in rankings.items():
            for rank, (item_id, raw_score) in enumerate(results, start=1):
                entry = fused.setdefault(item_id, {"id": item_id, "score": 0.0, "sources": [], "raw_scores": {}})
                entry["score"] = float(entry["score"]) + 1.0 / (k + rank)
                entry["sources"].append(source)
                entry["raw_scores"][source] = raw_score
        return sorted(fused.values(), key=lambda item: float(item["score"]), reverse=True)

    @staticmethod
    def expand_neighbor_chunks(chunk_ids: list[str],
                               chunks_by_id: dict[str, Chunk]) -> list[str]:
        """为每个 chunk 扩展前后相邻块，保持顺序并去重。

        [c2] → [c1, c2, c3]（如果 c1=prev, c3=next）。
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            for candidate in [chunk.prev_chunk_id, chunk.chunk_id, chunk.next_chunk_id]:
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)
        return ordered

