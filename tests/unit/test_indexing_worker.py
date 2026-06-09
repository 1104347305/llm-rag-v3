import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.main.python.config import settings
from src.main.python.steps.indexing.worker import index_project
from src.main.python.db.local_store import LocalStore


class TestIndexingWorker(unittest.TestCase):
    def test_index_project_reuses_unchanged_files_without_reembedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "family.md").write_text("# 家庭医生服务\n\n家庭医生服务不限次。", encoding="utf-8")

            with patch("src.main.python.steps.indexing.worker.embed_text", return_value=[0.1, 0.2]), patch("src.main.python.steps.indexing.worker.ElasticsearchClient") as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                first = index_project("test-incremental-reuse", root, force=True, build_embeddings=True)

            with patch("src.main.python.steps.indexing.worker.embed_text", side_effect=AssertionError("embedding should be reused")), patch(
                "app.indexing.worker.ElasticsearchClient"
            ) as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                second = index_project("test-incremental-reuse", root, force=False, build_embeddings=True)

        assert first["files_reindexed"] == 1
        assert second["files_reused"] == 1
        assert second["files_reindexed"] == 0
        assert second["chunks_indexed"] == first["chunks_indexed"]

    def test_index_project_only_reembeds_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\n家庭医生服务不限次。", encoding="utf-8")
            (root / "b.md").write_text("# B\n\n门诊预约协助。", encoding="utf-8")
            project_id = "test-incremental-changed"

            with patch("src.main.python.steps.indexing.worker.embed_text", return_value=[0.1, 0.2]), patch("src.main.python.steps.indexing.worker.ElasticsearchClient") as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                index_project(project_id, root, force=True, build_embeddings=True)

            (root / "b.md").write_text("# B\n\n门诊预约协助可安排陪诊。", encoding="utf-8")
            with patch("src.main.python.steps.indexing.worker.embed_text", return_value=[0.3, 0.4]) as embed, patch(
                "app.indexing.worker.ElasticsearchClient"
            ) as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                result = index_project(project_id, root, force=False, build_embeddings=True)

        assert result["files_reused"] == 1
        assert result["files_reindexed"] == 1
        assert embed.call_count == 1

    def test_index_project_removes_deleted_files_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\n家庭医生服务不限次。", encoding="utf-8")
            (root / "b.md").write_text("# B\n\n门诊预约协助。", encoding="utf-8")
            project_id = "test-incremental-delete"

            with patch("src.main.python.steps.indexing.worker.ElasticsearchClient") as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                index_project(project_id, root, force=True, build_embeddings=False)

            (root / "b.md").unlink()
            with patch("src.main.python.steps.indexing.worker.ElasticsearchClient") as es_cls:
                es_cls.return_value.write_indexes.return_value = None
                result = index_project(project_id, root, force=False, build_embeddings=False)

            store = LocalStore()
            pages, chunks, _ = store.load_index(project_id)
            sqlite_chunk_count = store.chunk_count(project_id)
            fts_results = store.search_chunks_fts(project_id, "门诊预约协助", top_k=10)

        assert result["files_deleted"] == 1
        assert [page.path for page in pages] == ["a.md"]
        assert [chunk.path for chunk in chunks] == ["a.md"]
        assert sqlite_chunk_count == 1
        assert fts_results == []

    def test_index_project_does_not_write_elasticsearch_when_indexing_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "family.md").write_text("# 家庭医生服务\n\n家庭医生服务不限次。", encoding="utf-8")
            no_es_settings = replace(settings, es_url="http://localhost:9200", enable_es_indexing=False)
            with patch("src.main.python.steps.indexing.worker.settings", no_es_settings), patch("src.main.python.steps.indexing.worker.ElasticsearchClient") as es_cls:
                result = index_project("test-no-es-indexing", root, force=True, build_embeddings=False)

        es_cls.assert_not_called()
        assert result["elasticsearch_indexed"] is False
        assert result["elasticsearch_error"] is None


if __name__ == "__main__":
    unittest.main()
