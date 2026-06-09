import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.main.python.config import settings
from src.main.python.models import Chunk, Page
from src.main.python.steps.retrieval.context_builder import ContextBuilder
from src.main.python.steps.indexing.worker import index_project
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline
from src.main.python.db.local_store import LocalStore


class TestRetrieval(unittest.TestCase):
    def test_retrieve_context_from_sample_data(self):
        result = index_project("test-pingan", Path("data"), force=True)
        assert result["pages_indexed"] > 0
        with patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._bm25_search", return_value=[]):
            response = RetrievalPipeline.get().retrieve("test-pingan", "家庭医生服务次数", top_pages=3)
        assert response["pages"]
        assert "家庭医生" in response["pages_context"]

    def test_retrieve_context_can_disable_es_recall(self):
        index_project("test-pingan-no-es", Path("data"), force=True, build_embeddings=False)
        with patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._bm25_search") as bm25:
            response = RetrievalPipeline.get().retrieve(
                "test-pingan-no-es",
                "家庭医生服务次数",
                top_pages=3,
                include_es=False,
                include_vector=False,
                debug=True,
            )

        bm25.assert_not_called()
        assert response["pages"]
        assert response["debug"]["bm25"] == []
        assert response["debug"]["vector"] == []
        assert "家庭医生" in response["pages_context"]

    def test_retrieve_context_uses_sqlite_fts_when_es_is_disabled(self):
        index_project("test-pingan-fts", Path("data"), force=True, build_embeddings=False)

        with patch.object(LocalStore, "load_index", side_effect=AssertionError("full index should not be loaded")):
            response = RetrievalPipeline.get().retrieve(
                "test-pingan-fts",
                "家庭医生服务次数",
                top_pages=3,
                include_es=False,
                include_vector=False,
                debug=True,
            )

        assert response["retrieval_mode"] == "candidate_sqlite"
        assert response["debug"]["lexical"]
        assert "家庭医生" in response["pages_context"]

    def test_retrieve_context_can_disable_vector_recall(self):
        index_project("test-pingan-no-vector", Path("data"), force=True, build_embeddings=False)
        with patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._bm25_search", return_value=[]), patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._vector_search") as vector:
            response = RetrievalPipeline.get().retrieve(
                "test-pingan-no-vector",
                "家庭医生服务次数",
                top_pages=3,
                include_vector=False,
                debug=True,
            )

        vector.assert_not_called()
        assert response["pages"]
        assert response["debug"]["vector"] == []

    def test_retrieve_context_uses_sqlite_lsh_for_vector_recall(self):
        project_id = "test-pingan-vector-lsh"
        vector = [1.0, 0.0, 0.0, 0.0]
        with patch("src.main.python.steps.indexing.worker.embed_text", return_value=vector):
            index_project(project_id, Path("data"), force=True, build_embeddings=True)

        vector_settings = replace(settings, enable_vector_retrieval=True, vector_search_backend="sqlite_lsh", embedding_dim=4)
        with patch("src.main.python.steps.retrieval.pipeline.settings", vector_settings), patch("src.main.python.steps.retrieval.vector_retriever.settings", vector_settings), patch(
            "src.main.python.steps.retrieval.vector_retriever.embed_text", return_value=vector
        ), patch.object(LocalStore, "load_index", side_effect=AssertionError("full index should not be loaded")):
            response = RetrievalPipeline.get().retrieve(
                project_id,
                "家庭医生服务次数",
                top_pages=3,
                include_es=False,
                include_vector=True,
                debug=True,
            )

        assert response["retrieval_mode"] == "candidate_sqlite"
        assert response["debug"]["vector"]
        assert any("vector_sqlite_lsh_enabled" in reason for reason in response["fallback_reasons"])

    def test_retrieve_context_uses_candidate_sqlite_when_es_hits(self):
        index_project("test-pingan-candidate", Path("data"), force=True, build_embeddings=False)
        _, chunks, _ = LocalStore().load_index("test-pingan-candidate")
        hit = next(chunk for chunk in chunks if "家庭医生" in chunk.title or "家庭医生" in chunk.content)

        with patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._bm25_search", return_value=[(hit.chunk_id, 9.0)]), patch.object(
            LocalStore, "load_index", side_effect=AssertionError("full index should not be loaded")
        ):
            response = RetrievalPipeline.get().retrieve(
                "test-pingan-candidate",
                "家庭医生服务次数",
                top_pages=3,
                include_vector=False,
                debug=True,
            )

        assert response["retrieval_mode"] == "candidate_sqlite"
        assert response["pages"]
        assert "家庭医生" in response["pages_context"]

    def test_retrieve_context_skips_local_lexical_when_chunk_count_exceeds_limit(self):
        chunks = [
            Chunk(
                project_id="p",
                page_id=f"p{i}",
                chunk_id=f"p{i}#0000",
                path=f"p{i}.md",
                title=f"P{i}",
                heading_path=f"P{i}",
                type="page",
                sources=[],
                content="无关内容",
                chunk_index=0,
            )
            for i in range(3)
        ]
        pages = [
            Page(
                project_id="p",
                page_id=chunk.page_id,
                path=chunk.path,
                title=chunk.title,
                type="page",
                sources=[],
                wikilinks=[],
                content=chunk.content,
                metadata={},
                content_sha256="x",
                mtime=0,
            )
            for chunk in chunks
        ]

        small_limit_settings = replace(settings, enable_local_lexical_retrieval=True, local_lexical_max_chunks=2)
        with patch("src.main.python.steps.retrieval.pipeline.LocalStore") as store_cls, patch("src.main.python.steps.retrieval.pipeline.RetrievalPipeline._bm25_search", return_value=[]), patch(
            "app.retrieval.pipeline.lexical_search"
        ) as lexical, patch("src.main.python.steps.retrieval.pipeline.settings", small_limit_settings):
            store_cls.return_value.load_index.return_value = (pages, chunks, [])
            store_cls.return_value.has_sqlite_index.return_value = False
            store_cls.return_value.chunk_count.return_value = None
            response = RetrievalPipeline.get().retrieve("p", "无关", include_vector=False, debug=True)

        lexical.assert_not_called()
        assert response["debug"]["lexical"] == []
        assert any("lexical_fallback_skipped" in reason for reason in response["fallback_reasons"])

    def test_retrieve_context_falls_back_to_overview_when_no_hits(self):
        page = Page(
            project_id="p",
            page_id="overview",
            path="overview.md",
            title="Overview",
            type="overview",
            sources=[],
            wikilinks=[],
            content="# Overview\n\n这是项目总览。",
            metadata={},
            content_sha256="x",
            mtime=0,
        )
        chunk = Chunk(
            project_id="p",
            page_id=page.page_id,
            chunk_id="overview#0000",
            path=page.path,
            title=page.title,
            heading_path="Overview",
            type=page.type,
            sources=[],
            content=page.content,
            chunk_index=0,
        )

        with patch("src.main.python.steps.retrieval.pipeline.LocalStore") as store_cls:
            store_cls.return_value.load_index.return_value = ([page], [chunk], [])
            store_cls.return_value.has_sqlite_index.return_value = False
            store_cls.return_value.chunk_count.return_value = None
            response = RetrievalPipeline.get().retrieve("p", "完全不匹配的问题", include_es=False, include_vector=False)

        assert response["pages"][0]["source"] == "overview"
        assert "项目总览" in response["pages_context"]

    def test_context_builder_packs_page_body_not_only_matched_chunk(self):
        page = Page(
            project_id="p",
            page_id="family-doctor",
            path="entities/家庭医生服务.md",
            title="家庭医生服务",
            type="entity",
            sources=[],
            wikilinks=[],
            content="# 家庭医生服务\n\n" + ("前置说明。\n" * 200) + "\n## 服务次数\n\n家庭医生服务不限次。",
            metadata={},
            content_sha256="x",
            mtime=0,
        )
        chunk = Chunk(
            project_id="p",
            page_id=page.page_id,
            chunk_id="family-doctor#0000",
            path=page.path,
            title=page.title,
            heading_path="家庭医生服务",
            type=page.type,
            sources=[],
            content="# 家庭医生服务\n\n前置说明。",
            chunk_index=0,
        )

        context = ContextBuilder().build(
            [{"page_id": page.page_id, "score": 1.0, "chunk_ids": [chunk.chunk_id]}],
            {chunk.chunk_id: chunk},
            {page.page_id: page},
            max_context_size=204800,
        )

        assert "家庭医生服务不限次" in context["pages_context"]

    def test_context_builder_packs_matched_section_first_for_long_page(self):
        opening = "页面开头无关信息。\n" * 600
        target = "## 关键规则\n\n家庭医生服务不限次，适用对象为被保险人。"
        tail = "\n页面尾部无关信息。" * 600
        content = "# 长页面\n\n" + opening + target + tail
        page = Page(
            project_id="p",
            page_id="long-page",
            path="entities/long.md",
            title="长页面",
            type="entity",
            sources=[],
            wikilinks=[],
            content=content,
            metadata={},
            content_sha256="x",
            mtime=0,
        )
        chunk = Chunk(
            project_id="p",
            page_id=page.page_id,
            chunk_id="long-page#0008",
            path=page.path,
            title=page.title,
            heading_path="长页面 > 关键规则",
            type=page.type,
            sources=[],
            content=target,
            chunk_index=8,
        )

        context = ContextBuilder().build(
            [{"page_id": page.page_id, "score": 1.0, "chunk_ids": [chunk.chunk_id]}],
            {chunk.chunk_id: chunk},
            {page.page_id: page},
            max_context_size=12000,
        )

        assert "家庭医生服务不限次" in context["pages_context"]
