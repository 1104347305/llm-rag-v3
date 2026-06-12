import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.main.python.steps.indexing.worker import index_project
from src.main.python.steps.stores.local_store import LocalStore


class TestIndexingWorker(unittest.IsolatedAsyncioTestCase):
    async def test_index_project_reuses_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "family.md").write_text("# 家庭医生服务\n\n不限次。", encoding="utf-8")
            embeddings = AsyncMock(return_value=[[0.1, 0.2]])
            with patch(
                "src.main.python.steps.indexing.worker.EmbeddingBuilder.batch_embed",
                embeddings,
            ):
                first = await index_project(
                    "async-index-reuse", root, force=True, build_embeddings=True
                )
                second = await index_project(
                    "async-index-reuse", root, force=False, build_embeddings=True
                )

        self.assertEqual(first["files_reindexed"], 1)
        self.assertEqual(second["files_reused"], 1)
        self.assertEqual(embeddings.await_count, 1)

    async def test_index_project_removes_deleted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\n家庭医生服务。", encoding="utf-8")
            (root / "b.md").write_text("# B\n\n门诊预约协助。", encoding="utf-8")
            project_id = "async-index-delete"

            await index_project(project_id, root, force=True, build_embeddings=False)
            (root / "b.md").unlink()
            result = await index_project(
                project_id, root, force=False, build_embeddings=False
            )
            pages, chunks, _ = await LocalStore.get().load_index(project_id)
            fts_results = await LocalStore.get().search_chunks_fts(
                project_id, "门诊预约协助", top_k=10
            )

        self.assertEqual(result["files_deleted"], 1)
        self.assertEqual([page.path for page in pages], ["a.md"])
        self.assertEqual([chunk.path for chunk in chunks], ["a.md"])
        self.assertEqual(fts_results, [])
