import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from src.main.python.db.dashscope import DashScopeClient


class FakeOpenAI:
    calls = []
    content = "ok"

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({"api_key": self.api_key, "base_url": self.base_url, "kwargs": kwargs})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class TestDashScopeClient(unittest.TestCase):
    def test_embed_parses_openai_compatible_response(self):
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "embedding_model": "text-embedding-v4",
                "dashscope_embedding_base_url": "https://example.test/v1",
                "dashscope_rerank_base_url": "https://example.test/v1",
                "dashscope_chat_base_url": "https://example.test/v1",
                "dashscope_api_key": "sk-test",
            },
        )()
        with patch.object(client, "request", return_value={"data": [{"embedding": [0.1, 0.2]}]}):
            assert client.embed("衣服的质量杠杠的") == [0.1, 0.2]

    def test_rerank_parses_qwen_response(self):
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "rerank_model": "qwen3-rerank",
                "dashscope_embedding_base_url": "https://example.test/v1",
                "dashscope_rerank_base_url": "https://example.test/v1",
                "dashscope_chat_base_url": "https://example.test/v1",
                "dashscope_api_key": "sk-test",
            },
        )()
        with patch.object(client, "request", return_value={"results": [{"index": 2, "relevance_score": 0.88}]}):
            assert client.rerank("什么是重排序模型", ["a", "b", "c"], top_n=1)[0]["index"] == 2

    def test_chat_parses_qwen_response(self):
        FakeOpenAI.calls = []
        FakeOpenAI.content = "我是通义千问。"
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "llm_model": "qwen-plus",
                "llm_reasoning_effort": "",
                "dashscope_embedding_base_url": "https://example.test/v1",
                "dashscope_rerank_base_url": "https://example.test/v1",
                "dashscope_chat_base_url": "https://example.test/v1",
                "dashscope_api_key": "sk-test",
            },
        )()
        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai_module}), patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-test"}):
            assert client.chat([{"role": "user", "content": "你是谁？"}]) == "我是通义千问。"

    def test_chat_sends_reasoning_effort_when_configured(self):
        FakeOpenAI.calls = []
        FakeOpenAI.content = "9.11大。"
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "llm_model": "vanchin/deepseek-v4-pro",
                "llm_reasoning_effort": "high",
                "dashscope_embedding_base_url": "https://example.test/v1",
                "dashscope_rerank_base_url": "https://example.test/v1",
                "dashscope_chat_base_url": "https://example.test/v1",
                "dashscope_api_key": "sk-test",
            },
        )()
        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai_module}), patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-test"}):
            assert client.chat([{"role": "user", "content": "9.9和9.11哪个大"}]) == "9.11大。"

        call = FakeOpenAI.calls[0]
        body = call["kwargs"]
        assert call["api_key"] == "sk-test"
        assert call["base_url"] == "https://example.test/v1"
        assert body["model"] == "vanchin/deepseek-v4-pro"
        assert body["reasoning_effort"] == "high"
        assert "temperature" not in body

    def test_chat_sends_temperature_only_when_explicit(self):
        FakeOpenAI.calls = []
        FakeOpenAI.content = "ok"
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "llm_model": "qwen-plus",
                "llm_reasoning_effort": "",
                "dashscope_chat_base_url": "https://example.test/v1",
            },
        )()
        fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_openai_module}), patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-test"}):
            assert client.chat([{"role": "user", "content": "test"}], temperature=0.2) == "ok"

        body = FakeOpenAI.calls[0]["kwargs"]
        assert body["temperature"] == 0.2

    def test_chat_requires_dashscope_api_key(self):
        client = DashScopeClient.__new__(DashScopeClient)
        client.config = type(
            "Config",
            (),
            {
                "llm_model": "vanchin/deepseek-v4-pro",
                "llm_reasoning_effort": "high",
                "dashscope_chat_base_url": "https://example.test/v1",
                "dashscope_api_key": "",
            },
        )()

        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(Exception, "DASHSCOPE_API_KEY"):
            client.chat([{"role": "user", "content": "test"}])
