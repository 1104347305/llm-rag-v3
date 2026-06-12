# ============================================================
# PostgreSQL pgvector 向量存储
# ============================================================
# 数据模型:
#   单表 rag_vectors(project_id, chunk_id, page_id, path, vector)
#   vector 列类型: halfvec (半精度浮点，内存减半)
#   索引: HNSW (m=16, ef_construction=200)
#
# 向量格式:
#   halfvec 范围 [-65504, 65504]，超出自动截断
#   cosine 距离用 <=> 运算符: similarity = 1 - distance
#
# 依赖: psycopg2, PostgreSQL 16+ + pgvector 扩展
# ============================================================

from __future__ import annotations

import asyncio

import threading
from contextlib import contextmanager

from src.main.python.config import settings
from src.main.python.models import Chunk
from src.main.python.utils.logging import get_logger, log_event

logger = get_logger(__name__)


class PgVectorUnavailable(Exception):
    """pgvector 不可用异常（连接失败 / 扩展未安装 / psycopg2 未安装）。"""


class PgVectorStore:
    """PostgreSQL pgvector 向量存储。

    核心功能:
      - init_schema(): 建表 + 创建 pgvector 扩展
      - ensure_index(): 创建 HNSW 索引
      - upsert_vectors(): 批量写入 / 更新向量
      - search_vectors(): cosine 相似度搜索
      - delete_by_paths(): 按文件路径删除（增量索引用）
    """

    _instance: PgVectorStore | None = None
    _pool: object | None = None
    _lock = threading.Lock()

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or settings.pg_dsn
        self._dim: int | None = None

    @classmethod
    def get(cls, dsn: str | None = None) -> PgVectorStore:
        """获取单例实例（线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(dsn=dsn)
        return cls._instance

    @classmethod
    def close(cls) -> None:
        """关闭连接池并重置单例（用于优雅关闭）。"""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
        cls._instance = None

    def _get_dim(self) -> int:
        """获取向量维度，未配置时默认 1024。"""
        if self._dim is not None:
            return self._dim
        self._dim = settings.embedding_dim or 1024
        return self._dim

    def _get_pool(self):
        """获取或创建连接池（双重检查锁）。连接失败时抛出 PgVectorUnavailable。"""
        from psycopg2 import pool as pgpool
        from psycopg2 import OperationalError

        cls = type(self)
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    try:
                        cls._pool = pgpool.ThreadedConnectionPool(
                            settings.pg_pool_min_size, settings.pg_pool_max_size, self._dsn,
                        )
                    except OperationalError as exc:
                        raise PgVectorUnavailable(f"pgvector connection failed: {exc}") from exc
        return cls._pool

    @contextmanager
    def _connect(self):
        """psycopg2 连接上下文管理器（从连接池获取，自动归还）。"""
        try:
            import psycopg2
        except ModuleNotFoundError as exc:
            raise PgVectorUnavailable("psycopg2 is not installed: pip install psycopg2-binary") from exc
        pool = self._get_pool()
        conn = pool.getconn()
        conn.autocommit = False
        try:
            yield conn
        except psycopg2.Error as exc:
            conn.rollback()
            raise PgVectorUnavailable(f"PostgreSQL operation failed: {exc}") from exc
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

    async def init_schema(self) -> None:
        """初始化库表：创建 pgvector 扩展和 rag_vectors 表。

        表结构:
          project_id text  -- 项目标识
          chunk_id text    -- 块标识（与 project_id 组成联合主键）
          page_id text     -- 所属页面
          path text        -- 源文件路径
          vector halfvec   -- 半精度向量
        """
        await asyncio.to_thread(self._init_schema_sync)

    def _init_schema_sync(self) -> None:
        dim = self._get_dim()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("create extension if not exists vector")
                cur.execute("""
                    create table if not exists rag_vectors (
                        project_id text not null,
                        chunk_id text not null,
                        page_id text not null,
                        path text not null,
                        vector halfvec(%s) not null,
                        primary key (project_id, chunk_id)
                    )
                """, (dim,))
                cur.execute("create index if not exists idx_rag_vectors_project_id on rag_vectors (project_id)")
                cur.execute("create index if not exists idx_rag_vectors_path on rag_vectors (project_id, path)")
            conn.commit()
        log_event(logger, 20, "pgvector.schema_initialized", "pgvector schema created", dim=dim)

    async def ensure_index(self) -> None:
        """创建 HNSW 索引（如不存在）。

        HNSW 参数:
          m = 16              -- 每个节点的最大连接数
          ef_construction = 200 -- 构建时的搜索深度
          ef_search            -- 查询时搜索深度，由 settings.pg_vector_ef_search 控制
        """
        await asyncio.to_thread(self._ensure_index_sync)

    def _ensure_index_sync(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("select indexname from pg_indexes where indexname = 'idx_rag_vectors_hnsw'")
                if cur.fetchone() is None:
                    cur.execute("""
                        create index idx_rag_vectors_hnsw on rag_vectors
                        using hnsw (vector halfvec_cosine_ops)
                        with (m = 16, ef_construction = 200)
                    """)
                    conn.commit()
                    log_event(logger, 20, "pgvector.hnsw_created", "HNSW index created")

    async def upsert_vectors(self, project_id: str, chunks: list[Chunk]) -> None:
        """批量写入 / 更新向量（execute_values 批量操作）。"""
        await asyncio.to_thread(self._upsert_vectors_sync, project_id, chunks)

    def _upsert_vectors_sync(self, project_id: str, chunks: list[Chunk]) -> None:
        valid = [(project_id, c.chunk_id, c.page_id, c.path, self._vector_to_str(c.vector))
                 for c in chunks if c.vector]
        if not valid:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    from psycopg2.extras import execute_values
                except ImportError:
                    # fallback: 逐行写入
                    for row in valid:
                        cur.execute("""
                            insert into rag_vectors (project_id, chunk_id, page_id, path, vector)
                            values (%s, %s, %s, %s, %s::halfvec)
                            on conflict (project_id, chunk_id) do update
                            set page_id = excluded.page_id, path = excluded.path, vector = excluded.vector
                        """, row)
                else:
                    execute_values(cur, """
                        insert into rag_vectors (project_id, chunk_id, page_id, path, vector)
                        values %s
                        on conflict (project_id, chunk_id) do update
                        set page_id = excluded.page_id, path = excluded.path, vector = excluded.vector
                    """, valid, template="(%s, %s, %s, %s, %s::halfvec)")
            conn.commit()
        log_event(logger, 20, "pgvector.vectors_upserted", "vectors upserted to pgvector", count=len(valid))

    async def search_vectors(self, project_id: str, query_vector: list[float], top_k: int = 100) -> list[tuple[str, float]]:
        """cosine 相似度搜索。

        pgvector 的 <=> 返回 cosine 距离 (0~2)，转成相似度：similarity = 1 - distance。
        执行前设置 hnsw.ef_search 来平衡速度与精度。

        返回值: [(chunk_id, similarity), ...] 按相似度降序排列
        """
        return await asyncio.to_thread(self._search_vectors_sync, project_id, query_vector, top_k)

    def _search_vectors_sync(self, project_id: str, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if not query_vector:
            return []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("set local hnsw.ef_search = %s", (settings.pg_vector_ef_search,))
                cur.execute("""
                    select chunk_id, 1 - (vector <=> %s::halfvec) as similarity
                    from rag_vectors
                    where project_id = %s
                    order by vector <=> %s::halfvec
                    limit %s
                """, (self._vector_to_str(query_vector), project_id, self._vector_to_str(query_vector), top_k))
                rows = cur.fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    async def delete_by_chunk_ids(self, project_id: str, chunk_ids: list[str]) -> None:
        """按 chunk_id 批量删除向量。"""
        await asyncio.to_thread(self._delete_by_chunk_ids_sync, project_id, chunk_ids)

    def _delete_by_chunk_ids_sync(self, project_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "delete from rag_vectors where project_id = %s and chunk_id = any(%s)",
                    (project_id, chunk_ids),
                )
            conn.commit()

    async def delete_by_paths(self, project_id: str, paths: list[str]) -> None:
        """按文件路径批量删除向量（增量索引时清理旧数据）。"""
        await asyncio.to_thread(self._delete_by_paths_sync, project_id, paths)

    def _delete_by_paths_sync(self, project_id: str, paths: list[str]) -> None:
        if not paths:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "delete from rag_vectors where project_id = %s and path = any(%s)",
                    (project_id, paths),
                )
            conn.commit()

    async def chunk_count(self, project_id: str) -> int:
        """返回项目的向量总数。"""
        return await asyncio.to_thread(self._chunk_count_sync, project_id)

    def _chunk_count_sync(self, project_id: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("select count(*) from rag_vectors where project_id = %s", (project_id,))
                row = cur.fetchone()
        return int(row[0]) if row else 0

    async def drop_project(self, project_id: str) -> None:
        """删除项目所有向量。"""
        await asyncio.to_thread(self._drop_project_sync, project_id)

    def _drop_project_sync(self, project_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("delete from rag_vectors where project_id = %s", (project_id,))
            conn.commit()


    def _vector_to_str(self, vector: list[float]) -> str:
        return "[" + ",".join(str(max(min(v, 65504.0), -65504.0)) for v in vector) + "]"
