import unittest
from unittest.mock import AsyncMock

from src.main.python.steps.retrieval.bm25 import bm25_search


class TestBM25Elasticsearch(unittest.IsolatedAsyncioTestCase):
    async def test_bm25_delegates_to_elasticsearch(self):
        client = AsyncMock()
        client.search_chunks.return_value = [("chunk-a", 3.14)]

        results = await bm25_search("家庭医生", "pingan", 10, client=client)

        self.assertEqual(results, [("chunk-a", 3.14)])
        client.search_chunks.assert_awaited_once_with("pingan", "家庭医生", 10)
