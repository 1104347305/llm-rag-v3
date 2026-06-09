from __future__ import annotations

import unittest
from pathlib import Path

from src.main.python.eval.collect_rag_answer_results import add_timestamp


class OutputPathTest(unittest.TestCase):
    def test_add_timestamp_before_extension(self) -> None:
        self.assertEqual(
            add_timestamp(
                Path("docs/评估集_result.xlsx"),
                "20260607_150102",
            ),
            Path("docs/评估集_result_20260607_150102.xlsx"),
        )

    def test_add_timestamp_to_jsonl_path(self) -> None:
        self.assertEqual(
            add_timestamp(
                Path("outputs/rag_answer_results.jsonl"),
                "20260607_150102",
            ),
            Path("outputs/rag_answer_results_20260607_150102.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
