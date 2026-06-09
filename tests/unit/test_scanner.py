import tempfile
import unittest
from pathlib import Path

from src.main.python.steps.indexing.scanner import scan_markdown_files


class TestScanner(unittest.TestCase):
    def test_scan_prefers_data_when_project_root_is_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "docs").mkdir()
            (root / "data" / "kept.md").write_text("# Kept", encoding="utf-8")
            (root / "docs" / "ignored.md").write_text("# Ignored", encoding="utf-8")
            (root / "README.md").write_text("# Ignored", encoding="utf-8")

            files = scan_markdown_files(root)

        assert [path.name for path in files] == ["kept.md"]

    def test_scan_prefers_wiki_over_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wiki").mkdir()
            (root / "data").mkdir()
            (root / "wiki" / "wiki-page.md").write_text("# Wiki", encoding="utf-8")
            (root / "data" / "data-page.md").write_text("# Data", encoding="utf-8")

            files = scan_markdown_files(root)

        assert [path.name for path in files] == ["wiki-page.md"]


if __name__ == "__main__":
    unittest.main()
