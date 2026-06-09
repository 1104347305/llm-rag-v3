from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.main.python.eval import collect_llm_wiki_chat_stream_results as collector


class FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size=None):
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class CollectorTest(unittest.TestCase):
    def test_add_timestamp_before_extension(self) -> None:
        timestamped = collector.add_timestamp(
            Path("docs/评估集_result.xlsx"),
            "20260607_142530",
        )

        self.assertEqual(
            timestamped,
            Path("docs/评估集_result_20260607_142530.xlsx"),
        )

    def test_chat_stream_parses_split_utf8_sse_chunks(self) -> None:
        meta = {
            "type": "chat_meta",
            "retrieval_ms": 12,
            "retrieval_query": "盛世优享26 险种代码",
            "sources": [{"number": 1, "title": "投保规则", "path": "/rules.md", "score": 0.9}],
        }
        token = {"choices": [{"delta": {"content": "答案是1822。"}}]}
        payload = (
            f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        split_at = payload.index("答".encode("utf-8")) + 1
        response = FakeResponse([payload[:split_at], payload[split_at:]])
        fake_requests = SimpleNamespace(
            post=lambda *args, **kwargs: response,
            RequestException=Exception,
        )

        with patch.object(collector, "requests", fake_requests):
            result = collector.chat_stream("险种代码是多少")

        self.assertEqual(result["answer"], "答案是1822。")
        self.assertEqual(result["retrieval_ms"], 12)
        self.assertEqual(result["sources"][0]["title"], "投保规则")
        self.assertTrue(response.closed)

    def test_build_result_row_keeps_evaluation_fields(self) -> None:
        row = collector.build_result_row(
            3,
            {
                "question": "问题",
                "gold_answer": "标准答案",
                "category": "简单查询类",
                "previous_actual_answer": "旧答案",
            },
            {
                "answer": "新答案",
                "sources": [{"number": 1, "title": "来源", "path": "/a.md", "score": 0.8}],
                "retrieval_query": "检索词",
                "retrieval_ms": 25,
                "error": "",
            },
            100.5,
        )

        self.assertEqual(row["question"], "问题")
        self.assertEqual(row["actual_answer"], "新答案")
        self.assertEqual(row["gold_answer"], "标准答案")
        self.assertEqual(row["category"], "简单查询类")
        self.assertEqual(row["source_count"], 1)
        self.assertEqual(row["contexts"], ["[1] 来源 | path=/a.md | score=0.8"])
        self.assertTrue(row["ok"])


if __name__ == "__main__":
    unittest.main()
