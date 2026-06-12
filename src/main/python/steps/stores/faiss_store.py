# ============================================================
# FAISS 向量存储
# ============================================================
# 存储结构:
#   .rag/indexes/{project_id}.faiss            — FAISS 索引二进制文件
#   .rag/indexes/{project_id}.faiss_meta.json  — 元数据
#
# 增量更新策略（软删除 + 定期压缩）:
#   upsert: 旧 ID → 加入 _deleted_ids → 新向量用新 ID 添加
#   delete: chunk_id → 对应 faiss_id → 加入 _deleted_ids
#   search: 多取 top_k*2 条结果，过滤掉 _deleted_ids
#   compact: 当 deleted_ids 占比 > 20% 时自动重建
#
# 异步: 所有公共方法均为 async，CPU 密集操作通过 asyncio.to_thread 执行
#
# 依赖: faiss-cpu, numpy
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.main.python.config import settings
from src.main.python.models import Chunk
from loguru import logger

_COMPACT_THRESHOLD = 0.2


class FaissUnavailable(Exception):
    """FAISS 不可用异常。"""


class FaissStore:

    def __init__(self) -> None:
        self._indexes: dict[str, object] = {}
        self._id_to_meta: dict[str, dict[int, dict]] = {}
        self._chunk_to_id: dict[str, dict[str, int]] = {}
        self._deleted_ids: dict[str, set[int]] = {}
        self._next_ids: dict[str, int] = {}
        self._project_locks: dict[str, threading.RLock] = {}
        self._project_locks_guard = threading.Lock()

    def _project_lock(self, project_id: str) -> threading.RLock:
        with self._project_locks_guard:
            return self._project_locks.setdefault(project_id, threading.RLock())

    @staticmethod
    def _index_dir() -> Path:
        return settings.storage_dir / "indexes"

    @staticmethod
    def _index_path(project_id: str) -> Path:
        return FaissStore._index_dir() / f"{project_id}.faiss"

    @staticmethod
    def _meta_path(project_id: str) -> Path:
        return FaissStore._index_dir() / f"{project_id}.faiss_meta.json"

    def init_index(self, dim: int) -> None:
        try:
            import faiss  # noqa: F401
        except ModuleNotFoundError as exc:
            raise FaissUnavailable("faiss-cpu is not installed") from exc
        self._dim = dim

    # ── 内部同步方法 ──────────────────────────────────────

    def _ensure_loaded(self, project_id: str) -> bool:
        import faiss
        if project_id in self._indexes:
            return True
        index_path = self._index_path(project_id)
        if not index_path.exists():
            return False
        self._indexes[project_id] = faiss.read_index(str(index_path))
        meta_path = self._meta_path(project_id)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._id_to_meta[project_id] = {int(k): v for k, v in data.get("id_to_meta", {}).items()}
            self._chunk_to_id[project_id] = data.get("chunk_id_to_id", {})
            self._deleted_ids[project_id] = {int(x) for x in data.get("deleted_ids", [])}
            self._next_ids[project_id] = data.get("next_id", 0)
        return True

    def _ensure_new(self, project_id: str) -> None:
        import faiss
        if project_id not in self._indexes:
            dim = getattr(self, "_dim", 1024)
            self._indexes[project_id] = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            self._id_to_meta[project_id] = {}
            self._chunk_to_id[project_id] = {}
            self._deleted_ids[project_id] = set()
            self._next_ids[project_id] = 0

    def _save_index(self, project_id: str) -> None:
        import faiss
        index = self._indexes.get(project_id)
        if index is None:
            return
        index_dir = self._index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_path(project_id)
        temp_path = index_path.with_name(
            f".{index_path.name}.{uuid4().hex}.tmp"
        )
        try:
            faiss.write_index(index, str(temp_path))
            os.replace(temp_path, index_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        self._save_meta(project_id)

    def _save_meta(self, project_id: str) -> None:
        meta_path = self._meta_path(project_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_ids.get(project_id, 0),
            "id_to_meta": {str(k): v for k, v in self._id_to_meta.get(project_id, {}).items()},
            "chunk_id_to_id": self._chunk_to_id.get(project_id, {}),
            "deleted_ids": [int(x) for x in self._deleted_ids.get(project_id, set())],
        }
        temp_path = meta_path.with_name(
            f".{meta_path.name}.{uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, meta_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _active_count(self, project_id: str) -> int:
        index = self._indexes.get(project_id)
        if index is None:
            return 0
        return index.ntotal - len(self._deleted_ids.get(project_id, set()))

    # ── 异步公共方法 ──────────────────────────────────────

    async def load_index(self, project_id: str) -> bool:
        return await asyncio.to_thread(self._load_index_sync, project_id)

    def _load_index_sync(self, project_id: str) -> bool:
        with self._project_lock(project_id):
            return self._ensure_loaded(project_id)

    async def search_vectors(self, project_id: str, query_vector: list[float], top_k: int = 100) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._search_sync, project_id, query_vector, top_k)

    def _search_sync(self, project_id: str, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        with self._project_lock(project_id):
            return self._search_unlocked(project_id, query_vector, top_k)

    def _search_unlocked(self, project_id: str, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        import faiss
        if not query_vector:
            return []
        if not self._ensure_loaded(project_id):
            return []
        index = self._indexes[project_id]
        deleted_ids = self._deleted_ids.get(project_id, set())
        if index.ntotal == 0:
            return []
        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)
        fetch_k = min(top_k + len(deleted_ids) + 50, index.ntotal)
        scores, indices = index.search(q, fetch_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            faiss_id = int(idx)
            if faiss_id in deleted_ids:
                continue
            meta = self._id_to_meta[project_id].get(faiss_id)
            if meta is None:
                continue
            results.append((meta["chunk_id"], float(score)))
            if len(results) >= top_k:
                break
        return results

    async def upsert_vectors(self, project_id: str, chunks: list[Chunk]) -> None:
        return await asyncio.to_thread(self._upsert_sync, project_id, chunks)

    def _upsert_sync(self, project_id: str, chunks: list[Chunk]) -> None:
        with self._project_lock(project_id):
            self._upsert_unlocked(project_id, chunks)

    def _upsert_unlocked(self, project_id: str, chunks: list[Chunk]) -> None:
        import faiss
        valid = [c for c in chunks if c.vector]
        if not valid:
            return
        self._ensure_loaded(project_id)
        self._ensure_new(project_id)
        chunk_to_id = self._chunk_to_id[project_id]
        deleted_ids = self._deleted_ids[project_id]
        id_to_meta = self._id_to_meta[project_id]
        next_id = self._next_ids[project_id]
        index = self._indexes[project_id]

        for chunk in valid:
            old_id = chunk_to_id.get(chunk.chunk_id)
            if old_id is not None:
                deleted_ids.add(old_id)

        new_vectors, new_ids = [], []
        for chunk in valid:
            faiss_id = next_id
            next_id += 1
            new_vectors.append(np.array(chunk.vector, dtype=np.float32))
            new_ids.append(faiss_id)
            chunk_to_id[chunk.chunk_id] = faiss_id
            id_to_meta[faiss_id] = {"chunk_id": chunk.chunk_id, "page_id": chunk.page_id, "path": chunk.path}

        vectors_array = np.array(new_vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_array)
        index.add_with_ids(vectors_array, np.array(new_ids, dtype=np.int64))
        self._next_ids[project_id] = next_id

        active = self._active_count(project_id)
        if active > 0 and len(deleted_ids) / (active + len(deleted_ids)) > _COMPACT_THRESHOLD:
            self._compact_sync(project_id)
        else:
            self._save_index(project_id)

        logger.bind(event="faiss.vectors_upserted").info(
            "vectors upserted", project_id=project_id, upserted=len(valid),
            active=active, soft_deleted=len(deleted_ids))

    async def delete_by_paths(self, project_id: str, paths: list[str]) -> None:
        return await asyncio.to_thread(self._delete_paths_sync, project_id, paths)

    def _delete_paths_sync(self, project_id: str, paths: list[str]) -> None:
        with self._project_lock(project_id):
            self._delete_paths_unlocked(project_id, paths)

    def _delete_paths_unlocked(self, project_id: str, paths: list[str]) -> None:
        if not paths:
            return
        paths_set = set(paths)
        if not self._ensure_loaded(project_id):
            return
        chunk_ids = [
            meta["chunk_id"] for meta in self._id_to_meta[project_id].values()
            if meta.get("path") in paths_set
            and meta["chunk_id"] in self._chunk_to_id[project_id]
        ]
        if chunk_ids:
            self._delete_ids_sync(project_id, chunk_ids)

    async def delete_by_chunk_ids(self, project_id: str, chunk_ids: list[str]) -> None:
        return await asyncio.to_thread(self._delete_ids_sync, project_id, chunk_ids)

    def _delete_ids_sync(self, project_id: str, chunk_ids: list[str]) -> None:
        with self._project_lock(project_id):
            self._delete_ids_unlocked(project_id, chunk_ids)

    def _delete_ids_unlocked(self, project_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        if not self._ensure_loaded(project_id):
            return
        chunk_to_id = self._chunk_to_id[project_id]
        deleted_ids = self._deleted_ids[project_id]
        removed = 0
        for chunk_id in chunk_ids:
            faiss_id = chunk_to_id.pop(chunk_id, None)
            if faiss_id is not None:
                deleted_ids.add(faiss_id)
                removed += 1
        if removed == 0:
            return
        active = self._active_count(project_id)
        if active > 0 and len(deleted_ids) / (active + len(deleted_ids)) > _COMPACT_THRESHOLD:
            self._compact_sync(project_id)
        else:
            self._save_meta(project_id)
        logger.bind(event="faiss.vectors_deleted").info(
            "vectors soft-deleted", project_id=project_id,
            count=removed, soft_deleted_total=len(deleted_ids))

    async def compact(self, project_id: str) -> int:
        return await asyncio.to_thread(self._compact_sync, project_id)

    def _compact_sync(self, project_id: str) -> int:
        with self._project_lock(project_id):
            return self._compact_unlocked(project_id)

    def _compact_unlocked(self, project_id: str) -> int:
        import faiss
        id_to_meta = self._id_to_meta[project_id]
        deleted_ids = self._deleted_ids[project_id]
        old_index = self._indexes[project_id]
        dim = old_index.d

        kept = []
        for faiss_id, meta in id_to_meta.items():
            if faiss_id not in deleted_ids and meta["chunk_id"] in self._chunk_to_id.get(project_id, {}):
                vec = old_index.reconstruct(int(faiss_id))
                kept.append((faiss_id, vec, meta))

        new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        new_id_to_meta = {}
        new_chunk_to_id = {}
        new_next_id = 0

        if kept:
            vectors = np.array([v for _, v, _ in kept], dtype=np.float32)
            ids = np.array([i for i, _, _ in kept], dtype=np.int64)
            new_index.add_with_ids(vectors, ids)
            for faiss_id, _, meta in kept:
                new_id_to_meta[faiss_id] = meta
                new_chunk_to_id[meta["chunk_id"]] = faiss_id
            new_next_id = max(new_id_to_meta.keys()) + 1

        removed = len(deleted_ids)
        self._indexes[project_id] = new_index
        self._id_to_meta[project_id] = new_id_to_meta
        self._chunk_to_id[project_id] = new_chunk_to_id
        self._deleted_ids[project_id] = set()
        self._next_ids[project_id] = new_next_id
        self._save_index(project_id)
        logger.bind(event="faiss.compacted").info(
            "index compacted", project_id=project_id,
            removed=removed, active=len(new_id_to_meta))
        return removed

    async def chunk_count(self, project_id: str) -> int:
        return await asyncio.to_thread(self._chunk_count_sync, project_id)

    def _chunk_count_sync(self, project_id: str) -> int:
        with self._project_lock(project_id):
            return self._active_count(project_id)

    async def drop_project(self, project_id: str) -> None:
        return await asyncio.to_thread(self._drop_sync, project_id)

    def _drop_sync(self, project_id: str) -> None:
        with self._project_lock(project_id):
            self._indexes.pop(project_id, None)
            self._id_to_meta.pop(project_id, None)
            self._chunk_to_id.pop(project_id, None)
            self._deleted_ids.pop(project_id, None)
            self._next_ids.pop(project_id, None)
            for p in [self._index_path(project_id), self._meta_path(project_id)]:
                if p.exists():
                    p.unlink()

    async def is_available(self, project_id: str) -> bool:
        return await asyncio.to_thread(
            lambda: self._index_path(project_id).exists() or project_id in self._indexes
        )
