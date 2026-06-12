import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.main.python.models import Chunk, Page
from src.main.python.steps.indexing.worker import index_project
from src.main.python.steps.retrieval.context_builder import ContextBuilder
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline


class TestRetrieval(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_context_uses_async_sqlite_fts(self):
        await index_project(
            "async-retrieval-fts", Path("data"), force=True, build_embeddings=False
        )
        pipeline = RetrievalPipeline()
        with patch.object(
            pipeline.reranker,
            "rerank_chunks",
            AsyncMock(return_value={}),
        ):
            response = await pipeline.retrieve(
                "async-retrieval-fts",
                "家庭医生服务次数",
                top_pages=3,
                include_es=False,
                include_vector=False,
                debug=True,
            )

        self.assertEqual(response["retrieval_mode"], "hybrid")
        self.assertTrue(response["debug"]["fts5"])
        self.assertIn("家庭医生", response["pages_context"])

    async def test_three_recall_paths_are_awaited(self):
        pipeline = RetrievalPipeline()
        pipeline.bm25_search = AsyncMock(return_value=[])
        pipeline.vector_search_store = AsyncMock(return_value=[])
        pipeline.lexicalRetriever.search = AsyncMock(return_value=[])
        pipeline.store.load_index = AsyncMock(side_effect=FileNotFoundError("missing"))

        with self.assertRaises(FileNotFoundError):
            await pipeline.retrieve(
                "missing", "query",
                include_es=True, include_vector=True, include_lexical=True,
            )

        pipeline.bm25_search.assert_awaited_once()
        pipeline.vector_search_store.assert_awaited_once()
        pipeline.lexicalRetriever.search.assert_awaited_once()

    async def test_falls_back_to_overview(self):
        page = Page(
            project_id="p", page_id="overview", path="overview.md",
            title="Overview", type="overview", sources=[], wikilinks=[],
            content="# Overview\n\n这是项目总览。", metadata={},
            content_sha256="x", mtime=0,
        )
        chunk = Chunk(
            project_id="p", page_id="overview", chunk_id="overview#0000",
            path="overview.md", title="Overview", heading_path="Overview",
            type="overview", sources=[], content=page.content, chunk_index=0,
        )
        pipeline = RetrievalPipeline()
        pipeline.store.load_index = AsyncMock(return_value=([page], [chunk], []))
        pipeline.lexicalRetriever.search = AsyncMock(return_value=[])

        response = await pipeline.retrieve(
            "p", "完全不匹配", include_es=False, include_vector=False
        )

        self.assertEqual(response["pages"][0]["source"], "overview")


class TestContextBuilder(unittest.TestCase):
    def test_packs_full_page_when_budget_allows(self):
        page = Page(
            project_id="p", page_id="family", path="family.md",
            title="家庭医生", type="entity", sources=[], wikilinks=[],
            content="# 家庭医生\n\n家庭医生服务不限次。", metadata={},
            content_sha256="x", mtime=0,
        )
        chunk = Chunk(
            project_id="p", page_id="family", chunk_id="family#0000",
            path="family.md", title="家庭医生", heading_path="家庭医生",
            type="entity", sources=[], content=page.content, chunk_index=0,
        )
        context = ContextBuilder().build(
            [{"page_id": "family", "score": 1.0, "chunk_ids": ["family#0000"]}],
            {"family#0000": chunk},
            {"family": page},
            10000,
        )
        self.assertIn("家庭医生服务不限次", context["pages_context"])
