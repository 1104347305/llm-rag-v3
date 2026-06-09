from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from src.main.python.config import Settings, settings

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class DashScopeUnavailable(RuntimeError):
    """Raised when DashScope is not configured or a request fails."""


class DashScopeClient:
    _instance: DashScopeClient | None = None
    _lock = threading.Lock()

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._openai_client: object | None = None

    @classmethod
    def get(cls, config: Settings | None = None) -> DashScopeClient:
        """获取单例实例（线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config or settings)
        return cls._instance

    def _get_openai_client(self, timeout: float | None = None) -> object:
        """缓存 OpenAI client，避免每次调用都新建连接。"""
        from openai import OpenAI
        if self._openai_client is None:
            api_key = os.getenv("DASHSCOPE_API_KEY") or self.config.dashscope_api_key
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=self.config.dashscope_chat_base_url,
                timeout=timeout or self.config.dashscope_request_timeout,
                max_retries=0,
            )
        return self._openai_client

    def embed(self, text: str) -> list[float]:
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        payload = self.request(
            self.config.dashscope_embedding_base_url,
            "embeddings",
            {"model": self.config.embedding_model, "input": text},
            api_key=self.config.dashscope_api_key,
        )
        data = payload.get("data") or []
        if not data:
            raise DashScopeUnavailable("DashScope embedding response has no data")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            raise DashScopeUnavailable("DashScope embedding response has no embedding")
        return [float(value) for value in embedding]

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入，一次 API 调用处理多条文本。"""
        if not texts:
            return []
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        payload = self.request(
            self.config.dashscope_embedding_base_url,
            "embeddings",
            {"model": self.config.embedding_model, "input": texts},
            api_key=self.config.dashscope_api_key,
        )
        data = payload.get("data") or []
        if not data:
            raise DashScopeUnavailable("DashScope embedding response has no data")
        results: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                results.append([float(v) for v in embedding])
            else:
                results.append([])
        return results

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict[str, Any]]:
        if not documents:
            return []
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        payload = self.request(
            self.config.dashscope_rerank_base_url,
            "reranks",
            {
                "model": self.config.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": top_n or len(documents),
            },
            api_key=self.config.dashscope_api_key,
        )
        results = payload.get("results", payload if isinstance(payload, list) else [])
        if not isinstance(results, list):
            raise DashScopeUnavailable("DashScope rerank response has no results")
        normalized = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score", 0.0))
            if index is not None:
                normalized.append({"index": int(index), "score": float(score), "raw": item})
        return normalized

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None, timeout: float | None = None, model: str | None = None, max_tokens: int | None = None) -> str:
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        try:
            client = self._get_openai_client(timeout)
            request: dict[str, Any] = {
                "model": model or self.config.llm_model,
                "messages": messages,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            if self.config.llm_reasoning_effort:
                request["reasoning_effort"] = self.config.llm_reasoning_effort
            if temperature is not None:
                request["temperature"] = temperature
            completion = client.chat.completions.create(**request)
        except Exception as exc:
            raise DashScopeUnavailable(
                f"DashScope chat unavailable: {exc}; chat_model={self.config.llm_model}; chat_base_url={self.config.dashscope_chat_base_url}"
            ) from exc
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise DashScopeUnavailable(
                f"DashScope chat response has no message content; chat_model={self.config.llm_model}; chat_base_url={self.config.dashscope_chat_base_url}"
            ) from exc
        if not isinstance(content, str):
            raise DashScopeUnavailable(
                f"DashScope chat response has no message content; chat_model={self.config.llm_model}; chat_base_url={self.config.dashscope_chat_base_url}"
            )
        return content

    def chat_stream(self, messages: list[dict[str, str]], temperature: float | None = None):
        """流式 LLM 调用，yield 文本块。"""
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        try:
            client = self._get_openai_client()
            request: dict[str, Any] = {"model": self.config.llm_model, "messages": messages, "stream": True}
            if self.config.llm_reasoning_effort:
                request["reasoning_effort"] = self.config.llm_reasoning_effort
            if temperature is not None:
                request["temperature"] = temperature
            stream = client.chat.completions.create(**request)
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            raise DashScopeUnavailable(f"DashScope stream failed: {exc}") from exc

    def request(self, base_url: str, path: str, body: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        max_retries = self.config.dashscope_max_retries
        backoff_base = self.config.dashscope_backoff_base
        timeout = self.config.dashscope_request_timeout

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                req = Request(
                    url,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urlopen(req, timeout=timeout) as response:
                    text = response.read().decode("utf-8")
                    return json.loads(text) if text else {}
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = DashScopeUnavailable(f"DashScope HTTP {exc.code}: {detail}")
                if exc.code not in _RETRYABLE_STATUSES or attempt >= max_retries:
                    raise last_error from exc
            except URLError as exc:
                last_error = DashScopeUnavailable(f"DashScope unavailable: {exc}")
                if attempt >= max_retries:
                    raise last_error from exc
            time.sleep(backoff_base * (2 ** attempt))
        raise last_error  # type: ignore[misc]


_dashscope_client: DashScopeClient | None = None


def get_dashscope_client() -> DashScopeClient:
    """获取 DashScopeClient 单例。"""
    global _dashscope_client
    if _dashscope_client is None:
        _dashscope_client = DashScopeClient()
    return _dashscope_client
