# ============================================================
# 索引入口 - IndexBuilder
# ============================================================
# 索引流程:
#
#   扫描 Markdown 文件 → 解析 frontmatter + 正文 → 分块 → 嵌入
#   → 图边构建 → 三层存储写入
#
# 存储层:
#   1. LocalStore (SQLite + JSON) — 始终写入
#   2. PgVectorStore (pgvector)   — enable_vector_retrieval 开启时
#   3. Elasticsearch              — enable_es_retrieval 开启时
#
# 增量索引:
#   - 通过 manifest 对比 sha256 / chunker_version / embedding_model 判断变化
#   - 未变文件复用旧 page/chunk，不重新嵌入
#   - 变化的 re-index，删除的从 SQLite 清理
#   - force=True 时全量重建
# ============================================================

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.main.python.config import settings
from src.main.python.steps.indexing.chunker import DocumentChunker
from src.main.python.steps.indexing.embedding_builder import EmbeddingBuilder
from src.main.python.steps.indexing.graph_builder import GraphBuilder
from src.main.python.steps.indexing.markdown_parser import MarkdownParser
from src.main.python.models import Chunk, Page
from src.main.python.db.elasticsearch import ElasticsearchClient, ElasticsearchUnavailable
from src.main.python.db.local_store import LocalStore
from src.main.python.db.pgvector_store import PgVectorStore, PgVectorUnavailable
from src.main.python.utils.hashing import sha256_text, stable_id
from src.main.python.utils.logging import get_logger, log_event

UTC = timezone.utc
logger = get_logger(__name__)


class IndexBuilder:
    """Markdown 知识库索引构建器。

    使用示例:
        builder = IndexBuilder()
        result = builder.build("my-project", Path("data"))
        result = builder.build("my-project", Path("data"), force=True)
    """

    def __init__(self) -> None:
        """初始化索引构建器。"""
        pass

    def build(self, project_id: str, project_path: Path,
              force: bool = False, build_embeddings: bool = True) -> dict[str, object]:
        """索引一个 Markdown 知识库项目。

        Args:
            project_id: 项目唯一标识。
            project_path: Markdown 文件根目录。
            force: True = 全量重建。
            build_embeddings: 是否调用 API 生成向量。

        Returns:
            dict 包含 pages_indexed, chunks_indexed, edges_indexed 等统计。
        """
        root = self._knowledge_root(project_path)
        store = LocalStore.get()
        old_manifest = store.load_manifest(project_id)
        log_event(logger, 20, "index.start", "index project started",
                  project_id=project_id, project_path=str(project_path), root=str(root),
                  force=force, build_embeddings=build_embeddings,
                  has_old_manifest=bool(old_manifest))

        # 加载旧索引用于增量复用
        old_pages_by_path, old_chunks_by_path = self._load_old_index(store, project_id, old_manifest, force)

        indexed_at = datetime.now(UTC).isoformat()
        pages: list[Page] = []
        chunks: list[Chunk] = []
        manifest_files: dict[str, dict[str, object]] = {}
        reused_files = 0
        reindexed_files = 0
        changed_paths: set[str] = set()

        chunker = DocumentChunker()
        files = self._scan_files(project_path)

        for path in tqdm(files, desc="索引中", unit="file", ncols=100):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            content_sha = sha256_text(text)
            stat = path.stat()

            # 增量复用检查
            old_file = old_manifest.get("files", {}).get(relative, {})
            old_page = old_pages_by_path.get(relative)
            old_page_chunks = old_chunks_by_path.get(relative, [])
            unchanged = (
                not force
                and old_file.get("sha256") == content_sha
                and old_manifest.get("chunker_version") == settings.chunker_version
                and old_manifest.get("embedding_model") == settings.embedding_model
            )
            can_reuse = (unchanged and old_page is not None and old_page_chunks
                         and (not build_embeddings or all(chunk.vector for chunk in old_page_chunks)))

            if can_reuse:
                page = replace(old_page, mtime=stat.st_mtime, content_sha256=content_sha)
                page_chunks = sorted(old_page_chunks, key=lambda chunk: chunk.chunk_index)
                pages.append(page)
                chunks.extend(page_chunks)
                reused_files += 1
                manifest_files[relative] = {
                    "sha256": content_sha, "mtime": stat.st_mtime, "chunk_count": len(page_chunks),
                    "indexed_at": old_file.get("indexed_at", indexed_at), "status": "unchanged",
                }
                continue

            # 解析 → 分块
            parsed = MarkdownParser.parse(text, fallback_title=path.stem)
            page_id = str(parsed.metadata.get("dedup_key") or stable_id(relative))
            page = Page(
                project_id=project_id, page_id=page_id, path=relative,
                title=parsed.title, type=parsed.page_type,
                sources=parsed.sources, wikilinks=parsed.wikilinks,
                content=parsed.body, metadata=parsed.metadata,
                content_sha256=content_sha, mtime=stat.st_mtime, indexed_at=indexed_at,
            )
            page_chunks = chunker.chunk(page)
            page.chunk_count = len(page_chunks)
            pages.append(page)
            chunks.extend(page_chunks)
            reindexed_files += 1
            changed_paths.add(relative)
            manifest_files[relative] = {
                "sha256": content_sha, "mtime": stat.st_mtime, "chunk_count": len(page_chunks),
                "indexed_at": indexed_at, "status": "indexed",
            }

        # 批量嵌入
        if build_embeddings:
            pending_chunks = [c for c in chunks if not c.vector]
            if pending_chunks:
                texts = [f"Title: {c.title}\nSection: {c.heading_path}\nContent: {c.content}"
                         for c in pending_chunks]
                vectors = EmbeddingBuilder.batch_embed(texts, dim=settings.embedding_dim)
                for chunk, vector in zip(pending_chunks, vectors):
                    chunk.vector = vector

        # 图边
        edges = GraphBuilder.build(project_id, pages)

        # 删除已不存在的文件
        deleted_paths = set(old_manifest.get("files", {})) - set(manifest_files)

        # 持久化
        store.save_index(project_id, pages, chunks, edges,
                         changed_paths=changed_paths, deleted_paths=deleted_paths,
                         rebuild_sqlite=force)
        store.save_manifest(project_id, self._build_manifest(project_id, manifest_files))

        # pgvector
        pg_indexed, pg_error = self._write_pgvector(project_id, chunks, build_embeddings, deleted_paths)

        # Elasticsearch
        es_indexed, es_error = self._write_elasticsearch(project_id, pages, chunks, edges,
                                                          changed_paths, deleted_paths, force)

        result = {
            "project_id": project_id,
            "pages_indexed": len(pages),
            "chunks_indexed": len(chunks),
            "edges_indexed": len(edges),
            "files_reused": reused_files,
            "files_reindexed": reindexed_files,
            "files_deleted": len(deleted_paths),
            "pgvector_indexed": pg_indexed,
            "pgvector_error": pg_error,
            "elasticsearch_indexed": es_indexed,
            "elasticsearch_error": es_error,
            "index_path": str(store.index_path(project_id)),
            "manifest_path": str(store.manifest_path(project_id)),
        }
        log_event(logger, 20, "index.completed", "index project completed", **result)
        return result

    # ── 文件扫描 ────────────────────────────────────────────

    @staticmethod
    def _scan_files(project_path: Path) -> list[Path]:
        """扫描 project_path 下所有 .md 文件，排除 .rag 目录。"""
        root = IndexBuilder._knowledge_root(project_path)
        return sorted(path for path in root.rglob("*.md")
                      if path.is_file() and ".rag" not in path.parts)

    @staticmethod
    def _knowledge_root(project_path: Path) -> Path:
        """自动检测知识库根目录（优先 wiki/ → data/ → project_path）。"""
        if (project_path / "wiki").is_dir():
            return project_path / "wiki"
        if (project_path / "data").is_dir():
            return project_path / "data"
        return project_path

    # ── 增量复用 ────────────────────────────────────────────

    def _load_old_index(self, store: LocalStore, project_id: str,
                         old_manifest: dict, force: bool) -> tuple[dict[str, Page], dict[str, list[Chunk]]]:
        """加载旧索引用于增量复用判断。"""
        if not old_manifest or force:
            return {}, {}
        try:
            old_pages, old_chunks, _ = store.load_index(project_id)
            old_pages_by_path = {page.path: page for page in old_pages}
            old_chunks_by_path: dict[str, list[Chunk]] = {}
            for chunk in old_chunks:
                old_chunks_by_path.setdefault(chunk.path, []).append(chunk)
            return old_pages_by_path, old_chunks_by_path
        except FileNotFoundError:
            return {}, {}

    # ── Manifest ────────────────────────────────────────────

    @staticmethod
    def _build_manifest(project_id: str, files: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """构建 manifest 元数据。"""
        return {
            "project_id": project_id,
            "chunker_version": settings.chunker_version,
            "embedding_model": settings.embedding_model,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }

    # ── 外部存储写入 ────────────────────────────────────────

    @staticmethod
    def _write_pgvector(project_id: str, chunks: list[Chunk],
                         build_embeddings: bool, deleted_paths: set[str]) -> tuple[bool, str | None]:
        """写入 pgvector（如启用且嵌入已构建）。"""
        if not (settings.pgvector_enabled and build_embeddings):
            if settings.pgvector_enabled:
                log_event(logger, 20, "index.pgvector_skipped",
                          "pgvector indexing skipped (build_embeddings disabled)", project_id=project_id)
            return False, None
        try:
            pg_store = PgVectorStore.get()
            pg_store.init_schema()
            pg_store.ensure_index()
            if deleted_paths:
                pg_store.delete_by_paths(project_id, list(deleted_paths))
            pg_store.upsert_vectors(project_id, [chunk for chunk in chunks if chunk.vector])
            return True, None
        except PgVectorUnavailable as exc:
            log_event(logger, 30, "index.pgvector_unavailable",
                      "pgvector indexing failed", project_id=project_id, error=str(exc))
            return False, str(exc)

    @staticmethod
    def _write_elasticsearch(project_id: str, pages: list[Page], chunks: list[Chunk],
                              edges: list, changed_paths: set[str], deleted_paths: set[str],
                              force: bool) -> tuple[bool, str | None]:
        """写入 Elasticsearch（如启用）。"""
        if not settings.es_indexing_enabled:
            log_event(logger, 20, "index.elasticsearch_skipped",
                      "elasticsearch indexing disabled", project_id=project_id)
            return False, None
        try:
            ElasticsearchClient.get().write_indexes(project_id, pages, chunks, edges,
                                                     changed_paths=changed_paths,
                                                     deleted_paths=deleted_paths, rebuild=force)
            return True, None
        except ElasticsearchUnavailable as exc:
            log_event(logger, 30, "index.elasticsearch_unavailable",
                      "elasticsearch indexing failed", project_id=project_id, error=str(exc))
            return False, str(exc)


# 向后兼容
def index_project(project_id: str, project_path: Path, force: bool = False,
                  build_embeddings: bool = True) -> dict[str, object]:
    """向后兼容入口（services/rag_service.py 使用）。"""
    return IndexBuilder().build(project_id, project_path, force, build_embeddings)
