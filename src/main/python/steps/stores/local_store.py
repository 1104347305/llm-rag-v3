from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
from uuid import uuid4
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.main.python.config import settings
from src.main.python.models import Chunk, GraphEdge, Page


class LocalStore:
    _instance: LocalStore | None = None
    _lock = threading.Lock()

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or settings.storage_dir
        self.index_dir = self.storage_dir / "indexes"
        self.manifest_dir = self.storage_dir / "manifests"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls, storage_dir: Path | None = None) -> LocalStore:
        """获取单例实例（线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(storage_dir=storage_dir)
        return cls._instance

    async def save_index(
        self, project_id: str, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge],
        changed_paths: set[str] | None = None, deleted_paths: set[str] | None = None,
        rebuild_sqlite: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._save_index_sync, project_id, pages, chunks, edges,
            changed_paths, deleted_paths, rebuild_sqlite,
        )

    def _save_index_sync(
        self, project_id: str, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge],
        changed_paths: set[str] | None, deleted_paths: set[str] | None,
        rebuild_sqlite: bool,
    ) -> None:
        payload = {
            "pages": [asdict(page) for page in pages],
            "chunks": [asdict(chunk) for chunk in chunks],
            "edges": [asdict(edge) for edge in edges],
        }
        atomic_write_text(
            self.index_path(project_id),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        if rebuild_sqlite or changed_paths is None or not self.sqlite_path(project_id).exists():
            self.save_sqlite_index(project_id, pages, chunks, edges)
        else:
            self.save_sqlite_incremental(project_id, pages, chunks, edges, changed_paths, deleted_paths or set())

    async def load_index(self, project_id: str) -> tuple[list[Page], list[Chunk], list[GraphEdge]]:
        """从 JSON 快照加载完整索引。"""
        return await asyncio.to_thread(self._load_index_sync, project_id)

    def _load_index_sync(self, project_id: str) -> tuple[list[Page], list[Chunk], list[GraphEdge]]:
        path = self.index_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"project index not found: {project_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            [Page(**item) for item in payload.get("pages", [])],
            [Chunk(**item) for item in payload.get("chunks", [])],
            [GraphEdge(**item) for item in payload.get("edges", [])],
        )

    async def save_manifest(self, project_id: str, manifest: dict[str, Any]) -> None:
        await asyncio.to_thread(
            atomic_write_text,
            self.manifest_path(project_id),
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    async def load_manifest(self, project_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._load_manifest_sync, project_id)

    def _load_manifest_sync(self, project_id: str) -> dict[str, Any]:
        path = self.manifest_path(project_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def index_path(self, project_id: str) -> Path:
        """获取索引 JSON 快照路径。"""
        return self.index_dir / f"{project_id}.json"

    def manifest_path(self, project_id: str) -> Path:
        """获取 manifest JSON 路径。"""
        return self.manifest_dir / f"{project_id}.json"

    def sqlite_path(self, project_id: str) -> Path:
        """获取 SQLite 数据库路径。"""
        return self.index_dir / f"{project_id}.sqlite3"

    async def has_sqlite_index(self, project_id: str) -> bool:
        """检查 SQLite 索引是否存在。"""
        return await asyncio.to_thread(self.sqlite_path(project_id).exists)

    async def chunk_count(self, project_id: str) -> int | None:
        return await asyncio.to_thread(self._chunk_count_sync, project_id)

    def _chunk_count_sync(self, project_id: str) -> int | None:
        if not self.sqlite_path(project_id).exists():
            return None
        with closing(self.connect(project_id)) as conn:
            row = conn.execute("select count(*) from chunks").fetchone()
        return int(row[0]) if row else 0

    async def load_chunks_by_ids(self, project_id: str, chunk_ids: list[str]) -> list[Chunk]:
        """按 chunk_id 列表从 SQLite 加载 Chunk。"""
        return await asyncio.to_thread(self._load_chunks_by_ids_sync, project_id, chunk_ids)

    def _load_chunks_by_ids_sync(self, project_id: str, chunk_ids: list[str]) -> list[Chunk]:
        ids = unique_strings(chunk_ids)
        if not ids:
            return []
        with closing(self.connect(project_id)) as conn:
            return [self.row_to_chunk(row) for row in select_by_ids(conn, "chunks", "chunk_id", ids)]

    async def load_pages_by_ids(self, project_id: str, page_ids: list[str]) -> list[Page]:
        """按 page_id 列表从 SQLite 加载 Page。"""
        return await asyncio.to_thread(self._load_pages_by_ids_sync, project_id, page_ids)

    def _load_pages_by_ids_sync(self, project_id: str, page_ids: list[str]) -> list[Page]:
        ids = unique_strings(page_ids)
        if not ids:
            return []
        with closing(self.connect(project_id)) as conn:
            return [self.row_to_page(row) for row in select_by_ids(conn, "pages", "page_id", ids)]

    async def load_chunks_for_pages(self, project_id: str, page_ids: list[str]) -> list[Chunk]:
        """加载指定页面的所有 chunk。"""
        return await asyncio.to_thread(self._load_chunks_for_pages_sync, project_id, page_ids)

    def _load_chunks_for_pages_sync(self, project_id: str, page_ids: list[str]) -> list[Chunk]:
        ids = unique_strings(page_ids)
        if not ids:
            return []
        with closing(self.connect(project_id)) as conn:
            rows = select_by_ids(conn, "chunks", "page_id", ids, order_by="page_id, chunk_index")
            return [self.row_to_chunk(row) for row in rows]

    async def load_edges_for_pages(self, project_id: str, page_ids: list[str]) -> list[GraphEdge]:
        """加载指定页面相关的所有图边。"""
        return await asyncio.to_thread(self._load_edges_for_pages_sync, project_id, page_ids)

    def _load_edges_for_pages_sync(self, project_id: str, page_ids: list[str]) -> list[GraphEdge]:
        ids = unique_strings(page_ids)
        if not ids:
            return []
        with closing(self.connect(project_id)) as conn:
            rows = select_by_ids(conn, "graph_edges", "source_page_id", ids)
            return [self.row_to_edge(row) for row in rows]

    async def search_chunks_fts(self, project_id: str, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """FTS5 全文搜索 chunks，返回 [(chunk_id, bm25_score), ...]。"""
        return await asyncio.to_thread(self._search_chunks_fts_sync, project_id, query, top_k)

    def _search_chunks_fts_sync(self, project_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
        fts_query = build_fts_query(query)
        if not fts_query or not self.sqlite_path(project_id).exists():
            return []
        try:
            with closing(self.connect(project_id)) as conn:
                rows = conn.execute("""
                    select chunk_id, bm25(chunk_fts, 0.0, 9.0, 6.0, 3.0, 2.0, 1.0, 4.0) as rank
                    from chunk_fts where chunk_fts match ? order by rank limit ?
                """, (fts_query, top_k)).fetchall()
        except sqlite3.Error:
            return []
        return [(str(row["chunk_id"]), -float(row["rank"])) for row in rows]

    def save_sqlite_index(self, project_id: str, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge]) -> None:
        with closing(self.connect(project_id)) as conn:
            self.create_sqlite_schema(conn, reset=True)
            self.insert_sqlite_records(conn, pages, chunks, edges)
            conn.commit()

    def save_sqlite_incremental(
        self, project_id: str, pages: list[Page], chunks: list[Chunk],
        edges: list[GraphEdge], changed_paths: set[str], deleted_paths: set[str],
    ) -> None:
        paths_to_replace = set(changed_paths) | set(deleted_paths)
        pages_to_insert = [page for page in pages if page.path in changed_paths]
        chunks_to_insert = [chunk for chunk in chunks if chunk.path in changed_paths]
        with closing(self.connect(project_id)) as conn:
            self.create_sqlite_schema(conn, reset=False)
            if paths_to_replace:
                old_chunk_ids = [str(row["chunk_id"]) for row in select_by_ids(conn, "chunks", "path", sorted(paths_to_replace))]
                delete_by_ids(conn, "chunk_fts", "chunk_id", old_chunk_ids)
                delete_by_ids(conn, "chunks", "path", sorted(paths_to_replace))
                delete_by_ids(conn, "pages", "path", sorted(paths_to_replace))
            conn.execute("delete from graph_edges where project_id = ?", (project_id,))
            self.insert_sqlite_records(conn, pages_to_insert, chunks_to_insert, edges)
            conn.commit()

    def create_sqlite_schema(self, conn: sqlite3.Connection, reset: bool) -> None:
        if reset:
            conn.executescript("""
                drop table if exists pages;
                drop table if exists chunks;
                drop table if exists graph_edges;
                drop table if exists chunk_fts;
            """)
        conn.executescript("""
            create table if not exists pages(
              project_id text not null, page_id text primary key,
              path text not null, title text not null, type text not null,
              sources text not null, wikilinks text not null,
              content text not null, metadata text not null,
              content_sha256 text not null, mtime real not null,
              chunk_count integer not null, indexed_at text not null
            );
            create table if not exists chunks(
              project_id text not null, page_id text not null,
              chunk_id text primary key, path text not null,
              title text not null, heading_path text not null,
              type text not null, sources text not null,
              content text not null, chunk_index integer not null,
              prev_chunk_id text, next_chunk_id text,
              token_estimate integer not null, vector text not null
            );
            create table if not exists graph_edges(
              project_id text not null, source_page_id text not null,
              target_page_id text not null, edge_type text not null,
              weight real not null
            );
            create index if not exists idx_chunks_page_id on chunks(page_id);
            create index if not exists idx_chunks_path on chunks(path);
            create index if not exists idx_pages_path on pages(path);
            create index if not exists idx_edges_source_page_id on graph_edges(source_page_id);
            create virtual table if not exists chunk_fts using fts5(
              chunk_id unindexed, title, heading_path, path, sources, content, terms,
              tokenize = 'unicode61'
            );
        """)

    def insert_sqlite_records(self, conn: sqlite3.Connection, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge]) -> None:
        conn.executemany(
            """insert or replace into pages values(
              :project_id, :page_id, :path, :title, :type, :sources, :wikilinks,
              :content, :metadata, :content_sha256, :mtime, :chunk_count, :indexed_at
            )""",
            [self.page_record(page) for page in pages],
        )
        conn.executemany(
            """insert or replace into chunks values(
              :project_id, :page_id, :chunk_id, :path, :title, :heading_path, :type,
              :sources, :content, :chunk_index, :prev_chunk_id, :next_chunk_id,
              :token_estimate, :vector
            )""",
            [self.chunk_record(chunk) for chunk in chunks],
        )
        conn.executemany(
            "insert into graph_edges values(:project_id, :source_page_id, :target_page_id, :edge_type, :weight)",
            [asdict(edge) for edge in edges],
        )
        conn.executemany(
            "insert into chunk_fts(chunk_id, title, heading_path, path, sources, content, terms) values(:chunk_id, :title, :heading_path, :path, :sources, :content, :terms)",
            [self.chunk_fts_record(chunk) for chunk in chunks],
        )

    def connect(self, project_id: str) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path(project_id))
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        conn.execute(f"pragma busy_timeout={settings.sqlite_busy_timeout}")
        return conn

    def page_record(self, page: Page) -> dict[str, object]:
        record = asdict(page)
        record["sources"] = json.dumps(page.sources, ensure_ascii=False)
        record["wikilinks"] = json.dumps(page.wikilinks, ensure_ascii=False)
        record["metadata"] = json.dumps(page.metadata, ensure_ascii=False)
        return record

    def chunk_record(self, chunk: Chunk) -> dict[str, object]:
        record = asdict(chunk)
        record["sources"] = json.dumps(chunk.sources, ensure_ascii=False)
        record["vector"] = json.dumps(chunk.vector, ensure_ascii=False)
        return record

    def chunk_fts_record(self, chunk: Chunk) -> dict[str, object]:
        return {
            "chunk_id": chunk.chunk_id, "title": chunk.title, "heading_path": chunk.heading_path,
            "path": chunk.path, "sources": " ".join(chunk.sources), "content": chunk.content,
            "terms": " ".join(search_terms(f"{chunk.title} {chunk.heading_path} {chunk.path} {' '.join(chunk.sources)} {chunk.content}")),
        }

    def row_to_page(self, row: sqlite3.Row) -> Page:
        data = dict(row)
        data["sources"] = json.loads(data["sources"])
        data["wikilinks"] = json.loads(data["wikilinks"])
        data["metadata"] = json.loads(data["metadata"])
        return Page(**data)

    def row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        data = dict(row)
        data["sources"] = json.loads(data["sources"])
        data["vector"] = json.loads(data["vector"])
        return Chunk(**data)

    def row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(**dict(row))


def unique_strings(values: list[str]) -> list[str]:
    """字符串去重保持顺序。"""
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def atomic_write_text(path: Path, content: str) -> None:
    """Write a complete file beside the target, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def select_by_ids(conn: sqlite3.Connection, table: str, column: str, ids: list[str], order_by: str | None = None) -> list[sqlite3.Row]:
    from src.main.python.config import settings
    batch_size = settings.sqlite_in_batch_size
    rows: list[sqlite3.Row] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start: start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        sql = f"select * from {table} where {column} in ({placeholders})"
        if order_by:
            sql += f" order by {order_by}"
        rows.extend(conn.execute(sql, batch).fetchall())
    return rows


def delete_by_ids(conn: sqlite3.Connection, table: str, column: str, ids: list[str]) -> None:
    from src.main.python.config import settings
    batch_size = settings.sqlite_in_batch_size
    for start in range(0, len(ids), batch_size):
        batch = ids[start: start + batch_size]
        if not batch:
            continue
        placeholders = ",".join("?" for _ in batch)
        conn.execute(f"delete from {table} where {column} in ({placeholders})", batch)


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
FTS_QUERY_RE = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)


def search_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in WORD_RE.findall(text.lower()):
        if len(word) > 1:
            terms.append(word)
    for match in CJK_RE.findall(text):
        chars = list(match)
        terms.extend(chars)
        terms.extend("".join(chars[i: i + 2]) for i in range(max(0, len(chars) - 1)))
        terms.extend("".join(chars[i: i + 3]) for i in range(max(0, len(chars) - 2)))
        if len(match) > 1:
            terms.append(match)
    return list(dict.fromkeys(term for term in terms if term.strip()))


def build_fts_query(query: str) -> str:
    terms = query_terms(query)
    if not terms:
        terms = [match.group(0).lower() for match in FTS_QUERY_RE.finditer(query)]
    if not terms:
        return ""
    from src.main.python.config import settings
    return " OR ".join(quote_fts_term(term) for term in terms[:settings.fts_max_query_terms])


def quote_fts_term(term: str) -> str:
    """对 FTS5 搜索词进行引号转义。"""
    return '"' + term.replace('"', '""') + '"'


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for word in WORD_RE.findall(query.lower()):
        if len(word) > 1:
            terms.append(word)
    for match in CJK_RE.findall(query):
        chars = list(match)
        if len(match) <= 2:
            terms.extend(chars)
        terms.extend("".join(chars[i: i + 2]) for i in range(max(0, len(chars) - 1)))
        terms.extend("".join(chars[i: i + 3]) for i in range(max(0, len(chars) - 2)))
        if len(match) > 1:
            terms.insert(0, match)
    return list(dict.fromkeys(term for term in terms if term.strip()))
