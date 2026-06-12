import unittest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from src.main.python.services.request_service import DashScopeClient, DashScopeUnavailable


class TestDashScopeClient(unittest.IsolatedAsyncioTestCase):
    def test_client_is_instantiated_with_async_openai(self):
        constructor = unittest.mock.Mock(return_value=object())
        with patch(
            "src.main.python.services.request_service.AsyncOpenAI",
            constructor,
        ):
            client = DashScopeClient(self.make_config())

        self.assertIs(client.client, constructor.return_value)
        constructor.assert_called_once()

    @staticmethod
    def make_config(
        api_key: str = "sk-test",
        embedding_url: str = "",
        rerank_url: str = "",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dashscope_api_key=api_key,
            dashscope_chat_base_url="https://example.test/v1",
            dashscope_embedding_base_url="https://example.test/v1",
            dashscope_rerank_base_url="https://example.test/v1",
            dashscope_request_timeout=5,
            dashscope_max_retries=0,
            embedding_model="text-embedding-v4",
            embedding_batch_size=10,
            rerank_model="qwen3-rerank",
            llm_model="qwen-plus",
            llm_reasoning_effort="",
            internal_embedding_url=embedding_url,
            internal_rerank_url=rerank_url,
            internal_model_timeout=5,
            internal_model_max_connections=8,
        )

    def make_client(
        self,
        api_key: str = "sk-test",
        embedding_url: str = "",
        rerank_url: str = "",
    ) -> DashScopeClient:
        with patch("src.main.python.services.request_service.AsyncOpenAI"):
            client = DashScopeClient(
                self.make_config(api_key, embedding_url, rerank_url)
            )
        client.client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock()),
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
        )
        return client

    async def test_embed_parses_response(self):
        client = self.make_client()
        client.client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
        self.assertEqual(await client.embed("文本"), [0.1, 0.2])

    async def test_chat_sends_reasoning_effort(self):
        client = self.make_client()
        client.config.llm_reasoning_effort = "high"
        client.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        self.assertEqual(await client.chat([{"role": "user", "content": "test"}]), "ok")
        body = client.client.chat.completions.create.await_args.kwargs
        self.assertEqual(body["reasoning_effort"], "high")

    async def test_chat_requires_api_key(self):
        client = self.make_client(api_key="")
        with self.assertRaisesRegex(DashScopeUnavailable, "DASHSCOPE_API_KEY"):
            await client.chat([{"role": "user", "content": "test"}])

    async def test_internal_embedding_posts_items_and_parses_vectors(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"embedding": [[0.1, 0.2], [0.3, 0.4]]},
            )

        client = self.make_client(
            api_key="",
            embedding_url="http://internal.test/embedding",
        )
        await client.http_client.aclose()
        client.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        result = await client.batch_embed(["a", "b"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(requests, [{"items": ["a", "b"]}])
        client.client.embeddings.create.assert_not_awaited()
        await client.http_client.aclose()

    async def test_internal_rerank_maps_scores_to_original_indexes(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "data": {
                        "query": "q",
                        "items": ["a", "b", "c"],
                        "scores": [0.2, 0.9, 0.5],
                    }
                },
            )

        client = self.make_client(
            api_key="",
            rerank_url="http://internal.test/rerank",
        )
        await client.http_client.aclose()
        client.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        result = await client.rerank("q", ["a", "b", "c"], top_n=2)

        self.assertEqual(
            result,
            [{"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}],
        )
        self.assertEqual(
            requests,
            [{"query": "q", "items": ["a", "b", "c"]}],
        )
        await client.http_client.aclose()

    async def test_internal_embedding_rejects_mismatched_count(self):
        client = self.make_client(
            embedding_url="http://internal.test/embedding",
        )
        await client.http_client.aclose()
        client.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"embedding": [[0.1]]}
                )
            )
        )

        with self.assertRaisesRegex(
            DashScopeUnavailable, "count does not match"
        ):
            await client.batch_embed(["a", "b"])
        await client.http_client.aclose()
