from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from urllib import error, request

from src.main.python.config import settings
from src.main.python.models import Chunk, GraphEdge, Page
from loguru import logger


_BULK_BATCH_SIZE = 500  # 已废弃，用 settings.es_bulk_batch_size
_ERROR_DETAIL_TRUNCATE = 500  # 错误详情截断长度


class ElasticsearchUnavailable(RuntimeError):
    """Raised when Elasticsearch is not configured or cannot serve a request."""


class ElasticsearchClient:
    _instance: ElasticsearchClient | None = None
    _opener: request.OpenerDirector | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not settings.es_url:
            raise ElasticsearchUnavailable("ES_URL is not configured")
        self._base = settings.es_url.rstrip("/")
        self._auth = (settings.es_user, settings.es_password) if settings.es_user else None

    @classmethod
    def get(cls) -> ElasticsearchClient:
        """获取单例实例（线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_opener(self) -> request.OpenerDirector:
        cls = type(self)
        if cls._opener is None:
            with cls._lock:
                if cls._opener is None:
                    cls._opener = request.build_opener(request.HTTPHandler())
        return cls._opener

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base}/{path.lstrip('/')}"
        data = json.dumps(body).encode("utf-8") if body else None
        headers = {"Connection": "keep-alive"}
        if data:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method)
        if self._auth:
            import base64
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{self._auth[0]}:{self._auth[1]}".encode()).decode())
        try:
            with self._get_opener().open(req, timeout=settings.es_search_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ElasticsearchUnavailable(f"Elasticsearch HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ElasticsearchUnavailable(f"Elasticsearch unavailable: {exc}") from exc

    async def search_chunks(self, project_id: str, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._search_chunks_sync, project_id, query, top_k)

    def _search_chunks_sync(self, project_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
        index = settings.es_chunks_index
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"project_id": project_id}},
                        {"match": {"content": {"query": query, "operator": "or"}}},
                    ],
                    "should": [
                        {"match": {"title": {"query": query, "boost": settings.es_title_boost}}},
                        {"match": {"heading_path": {"query": query, "boost": settings.es_heading_boost}}},
                    ],
                }
            },
        }
        result = self._request("POST", f"/{index}/_search", body)
        hits = result.get("hits", {}).get("hits", [])
        return [(hit["_source"]["chunk_id"], float(hit["_score"])) for hit in hits]

    async def write_indexes(
        self, project_id: str, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge],
        changed_paths: set[str] | None = None, deleted_paths: set[str] | None = None,
        rebuild: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._write_indexes_sync, project_id, pages, chunks, edges,
            changed_paths, deleted_paths, rebuild,
        )

    def _write_indexes_sync(
        self, project_id: str, pages: list[Page], chunks: list[Chunk], edges: list[GraphEdge],
        changed_paths: set[str] | None, deleted_paths: set[str] | None, rebuild: bool,
    ) -> None:
        if rebuild:
            self._recreate_index(settings.es_pages_index, self._pages_mapping())
            self._recreate_index(settings.es_chunks_index, self._chunks_mapping())
            self._recreate_index(settings.es_graph_edges_index, self._edges_mapping())
        self._bulk_index(settings.es_pages_index, [self._page_doc(page) for page in pages])
        self._bulk_index(settings.es_chunks_index, [self._chunk_doc(chunk) for chunk in chunks])
        self._bulk_index(settings.es_graph_edges_index, [self._edge_doc(edge) for edge in edges])

    def _raw_request(self, method: str, path: str, body_data: str) -> dict[str, Any]:
        """发送 raw body 请求（用于 bulk API 等 NDJSON 格式）。"""
        url = f"{self._base}/{path.lstrip('/')}"
        data = body_data.encode("utf-8")
        headers = {"Content-Type": "application/x-ndjson", "Connection": "keep-alive"}
        req = request.Request(url, data=data, headers=headers, method=method)
        if self._auth:
            import base64
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{self._auth[0]}:{self._auth[1]}".encode()).decode())
        try:
            with self._get_opener().open(req, timeout=settings.es_bulk_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ElasticsearchUnavailable(f"Elasticsearch HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ElasticsearchUnavailable(f"Elasticsearch unavailable: {exc}") from exc

    def _recreate_index(self, index: str, mapping: dict[str, Any]) -> None:
        try:
            self._request("DELETE", f"/{index}")
        except ElasticsearchUnavailable:
            pass
        self._request("PUT", f"/{index}", {"mappings": mapping})

    def _bulk_index(self, index: str, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        batch_size = settings.es_bulk_batch_size
        for batch_start in range(0, len(docs), batch_size):
            batch = docs[batch_start: batch_start + batch_size]
            lines: list[str] = []
            for doc in batch:
                lines.append(json.dumps({"index": {"_index": index, "_id": doc["_id"]}}, ensure_ascii=False))
                lines.append(json.dumps(doc["_source"], ensure_ascii=False))
            ndjson = "\n".join(lines) + "\n"
            result = self._raw_request("POST", "/_bulk", ndjson)
            if result.get("errors"):
                error_items = [item for item in result.get("items", []) if "error" in item.get("index", {})]
                raise ElasticsearchUnavailable(f"Elasticsearch bulk index returned errors: {error_items[:3]}")

    def _page_doc(self, page: Page) -> dict[str, Any]:
        return {
            "_id": f"{page.project_id}_{page.page_id}",
            "_source": {
                "project_id": page.project_id, "page_id": page.page_id,
                "path": page.path, "title": page.title, "type": page.type,
                "content": page.content,
            },
        }

    def _chunk_doc(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "_id": f"{chunk.project_id}_{chunk.chunk_id}",
            "_source": {
                "project_id": chunk.project_id, "page_id": chunk.page_id,
                "chunk_id": chunk.chunk_id, "path": chunk.path,
                "title": chunk.title, "heading_path": chunk.heading_path,
                "type": chunk.type, "content": chunk.content,
            },
        }

    def _edge_doc(self, edge: GraphEdge) -> dict[str, Any]:
        return {
            "_id": f"{edge.project_id}_{edge.source_page_id}_{edge.target_page_id}",
            "_source": {
                "project_id": edge.project_id,
                "source_page_id": edge.source_page_id,
                "target_page_id": edge.target_page_id,
                "edge_type": edge.edge_type, "weight": edge.weight,
            },
        }

    def _pages_mapping(self) -> dict[str, Any]:
        return {
            "properties": {
                "project_id": {"type": "keyword"}, "page_id": {"type": "keyword"},
                "path": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "type": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
            }
        }

    def _chunks_mapping(self) -> dict[str, Any]:
        return {
            "properties": {
                "project_id": {"type": "keyword"}, "page_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"}, "path": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "heading_path": {"type": "text", "analyzer": "standard"},
                "type": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
            }
        }

    def _edges_mapping(self) -> dict[str, Any]:
        return {
            "properties": {
                "project_id": {"type": "keyword"},
                "source_page_id": {"type": "keyword"},
                "target_page_id": {"type": "keyword"},
                "edge_type": {"type": "keyword"},
                "weight": {"type": "float"},
            }
        }
