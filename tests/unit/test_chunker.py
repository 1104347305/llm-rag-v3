import unittest

from src.main.python.steps.indexing.chunker import ChunkingConfig, chunk_page
from src.main.python.models import Page


class TestChunker(unittest.TestCase):
    def test_chunk_page_preserves_heading_path_and_neighbors(self):
        page = Page(
            project_id="p",
            page_id="family-doctor",
            path="entities/家庭医生服务.md",
            title="家庭医生服务",
            type="entity",
            sources=[],
            wikilinks=[],
            content="# 家庭医生服务\n\n## 服务定义\n\n这是服务定义。\n\n## 服务次数\n\n家庭不限次。",
            metadata={},
            content_sha256="x",
            mtime=0,
        )
        chunks = chunk_page(page)
        assert chunks
        assert "家庭医生服务" in chunks[0].heading_path
        assert chunks[0].prev_chunk_id is None

    def test_chunk_page_adds_overlap_between_chunks(self):
        first = "A" * 80
        second = "B" * 80
        page = Page(
            project_id="p",
            page_id="overlap",
            path="entities/overlap.md",
            title="Overlap",
            type="entity",
            sources=[],
            wikilinks=[],
            content=f"# Overlap\n\n{first}\n\n{second}",
            metadata={},
            content_sha256="x",
            mtime=0,
        )

        chunks = chunk_page(page, ChunkingConfig(target_chars=60, max_chars=100, min_chars=1, overlap_chars=10))

        assert len(chunks) >= 3
        assert chunks[2].content.startswith("A" * 10)
