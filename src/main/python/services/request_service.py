from __future__ import annotations

import asyncio
import builtins
from typing import Any, AsyncIterator

import httpx
from loguru import logger
from openai import AsyncOpenAI
from src.main.python.config import Settings, settings
from src.main.python.utils.concurrency import AsyncCapacityLimiter, ServiceOverloaded


class DashScopeUnavailable(RuntimeError):
    """Raised when DashScope is not configured or a request fails."""


class DashScopeClient:

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            api_key=self.config.dashscope_api_key,
            base_url=self.config.dashscope_chat_base_url,
            timeout=self.config.dashscope_request_timeout,
            max_retries=self.config.dashscope_max_retries,
        )
        self.http_client = httpx.AsyncClient(
            timeout=builtins.float(
                getattr(self.config, "internal_model_timeout", 30)
            ),
            limits=httpx.Limits(
                max_connections=builtins.int(
                    getattr(self.config, "internal_model_max_connections", 32)
                ),
                max_keepalive_connections=builtins.int(
                    getattr(self.config, "internal_model_max_connections", 32)
                ),
            ),
        )
        acquire_timeout = builtins.float(
            getattr(self.config, "overload_wait_seconds", 0.05)
        )
        self._chat_limiter = AsyncCapacityLimiter(
            "dashscope_chat",
            builtins.int(getattr(self.config, "max_concurrent_chat", 16)),
            acquire_timeout,
        )
        self._embedding_limiter = AsyncCapacityLimiter(
            "dashscope_embedding",
            builtins.int(
                getattr(self.config, "max_concurrent_embedding", 16)
            ),
            acquire_timeout,
        )
        self._rerank_limiter = AsyncCapacityLimiter(
            "dashscope_rerank",
            builtins.int(getattr(self.config, "max_concurrent_rerank", 8)),
            acquire_timeout,
        )
        logger.info(
            "model request backends initialized | embedding={} rerank={}",
            self._internal_embedding_url or "dashscope",
            self._internal_rerank_url or "dashscope",
        )

    def _get_client(self) -> AsyncOpenAI:
        return self.client

    # ── Embeddings ──────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        if self._internal_embedding_url:
            vectors = await self._internal_embed([text])
            return vectors[0]
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        client = self._get_client()
        try:
            async with self._embedding_limiter.slot():
                resp = await client.embeddings.create(
                    model=self.config.embedding_model, input=text,
                    extra_body={"base_url": self.config.dashscope_embedding_base_url})
        except ServiceOverloaded:
            raise
        except Exception as exc:
            raise DashScopeUnavailable(f"DashScope embed failed: {exc}") from exc
        data = resp.data
        if not data:
            raise DashScopeUnavailable("DashScope embedding response has no data")
        embedding = data[0].embedding
        if not isinstance(embedding, list):
            raise DashScopeUnavailable("DashScope embedding response has no embedding")
        return [builtins.float(v) for v in embedding]

    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._internal_embedding_url:
            return await self._internal_embed(texts)
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        client = self._get_client()
        results: list[list[float]] = []
        batch_size = self.config.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                async with self._embedding_limiter.slot():
                    resp = await client.embeddings.create(
                        model=self.config.embedding_model, input=batch)
                for item in resp.data:
                    emb = item.embedding
                    results.append(
                        [builtins.float(v) for v in emb]
                        if isinstance(emb, list)
                        else []
                    )
            except ServiceOverloaded:
                raise
            except Exception as exc:
                raise DashScopeUnavailable(f"DashScope batch embed failed: {exc}") from exc
        return results

    # ── Rerank ──────────────────────────────────────────────

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict[str, Any]]:
        if not documents:
            return []
        if self._internal_rerank_url:
            return await self._internal_rerank(query, documents, top_n)
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        # Rerank 使用独立的 base_url，不走 OpenAI 兼容接口
        async with self._rerank_limiter.slot():
            return await asyncio.to_thread(self._rerank_sync, query, documents, top_n)

    def _rerank_sync(self, query: str, documents: list[str], top_n: int | None) -> list[dict[str, Any]]:
        import json as _json
        from urllib.error import HTTPError
        from urllib.parse import urljoin
        from urllib.request import Request, urlopen

        url = urljoin(self.config.dashscope_rerank_base_url.rstrip("/") + "/", "reranks")
        body = _json.dumps({
            "model": self.config.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
        }, ensure_ascii=False).encode("utf-8")
        req = Request(url, data=body, headers={
            "Authorization": f"Bearer {self.config.dashscope_api_key}",
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(req, timeout=self.config.dashscope_request_timeout) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DashScopeUnavailable(f"DashScope rerank HTTP {exc.code}: {detail}") from exc
        except ServiceOverloaded:
            raise
        except Exception as exc:
            raise DashScopeUnavailable(f"DashScope rerank failed: {exc}") from exc

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
                normalized.append({
                    "index": builtins.int(index),
                    "score": builtins.float(score),
                    "raw": item,
                })
        return normalized

    @property
    def _internal_embedding_url(self) -> str:
        return str(getattr(self.config, "internal_embedding_url", "")).strip()

    @property
    def _internal_rerank_url(self) -> str:
        return str(getattr(self.config, "internal_rerank_url", "")).strip()

    async def _internal_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            async with self._embedding_limiter.slot():
                response = await self.http_client.post(
                    self._internal_embedding_url,
                    json={"items": texts},
                )
                response.raise_for_status()
                payload = response.json()
        except ServiceOverloaded:
            raise
        except Exception as exc:
            raise DashScopeUnavailable(
                f"Internal embedding request failed: {exc}"
            ) from exc

        embeddings = payload.get("embedding") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise DashScopeUnavailable(
                "Internal embedding response count does not match items"
            )

        normalized: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                raise DashScopeUnavailable(
                    "Internal embedding response contains an invalid vector"
                )
            try:
                normalized.append(
                    [builtins.float(value) for value in embedding]
                )
            except (TypeError, ValueError) as exc:
                raise DashScopeUnavailable(
                    "Internal embedding response contains a non-numeric vector"
                ) from exc
        return normalized

    async def _internal_rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None,
    ) -> list[dict[str, Any]]:
        try:
            async with self._rerank_limiter.slot():
                response = await self.http_client.post(
                    self._internal_rerank_url,
                    json={"query": query, "items": documents},
                )
                response.raise_for_status()
                payload = response.json()
        except ServiceOverloaded:
            raise
        except Exception as exc:
            raise DashScopeUnavailable(
                f"Internal rerank request failed: {exc}"
            ) from exc

        data = payload.get("data") if isinstance(payload, dict) else None
        scores = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(scores, list) or len(scores) != len(documents):
            raise DashScopeUnavailable(
                "Internal rerank response score count does not match items"
            )

        try:
            ranked = sorted(
                (
                    {"index": index, "score": builtins.float(score)}
                    for index, score in enumerate(scores)
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
        except (TypeError, ValueError) as exc:
            raise DashScopeUnavailable(
                "Internal rerank response contains a non-numeric score"
            ) from exc
        return ranked[: top_n or len(ranked)]

    # ── Chat ────────────────────────────────────────────────

    async def chat(self, messages: list[dict[str, str]],
                   model: str | None = None, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        _model = model or self.config.llm_model
        client = self._get_client()
        try:
            body: dict[str, Any] = {"model": _model, "messages": messages}
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature
            if self.config.llm_reasoning_effort:
                body["reasoning_effort"] = self.config.llm_reasoning_effort
            async with self._chat_limiter.slot():
                completion = await client.chat.completions.create(**body)
        except ServiceOverloaded:
            raise
        except Exception as exc:
            raise DashScopeUnavailable(f"DashScope chat failed: {exc}; model={_model}") from exc
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise DashScopeUnavailable(f"DashScope chat response empty; model={_model}")
        if not isinstance(content, str):
            raise DashScopeUnavailable(f"DashScope chat response empty; model={_model}")
        return content

    async def chat_stream(self, messages: list[dict[str, str]],
                          model: str | None = None, temperature: float | None = None):
        """异步流式 LLM 调用，yield 文本块。"""
        stream = await self.create_chat_stream(
            messages, model=model, temperature=temperature
        )
        async for text in stream:
            yield text

    async def create_chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """在返回响应头前获取并发槽并创建流，流结束时自动释放。"""
        if not self.config.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")
        _model = model or self.config.llm_model
        client = self._get_client()
        await self._chat_limiter.acquire()
        try:
            body: dict[str, Any] = {"model": _model, "messages": messages, "stream": True}
            if temperature is not None:
                body["temperature"] = temperature
            if self.config.llm_reasoning_effort:
                body["reasoning_effort"] = self.config.llm_reasoning_effort
            stream = await client.chat.completions.create(**body)
        except asyncio.CancelledError:
            self._chat_limiter.release()
            raise
        except Exception as exc:
            self._chat_limiter.release()
            raise DashScopeUnavailable(f"DashScope stream failed: {exc}; model={_model}") from exc

        async def _iterate() -> AsyncIterator[str]:
            try:
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content
            finally:
                self._chat_limiter.release()

        return _iterate()

    async def close(self) -> None:
        await self.http_client.aclose()
        await self.client.close()


_dashscope_client: DashScopeClient | None = None


def get_dashscope_client() -> DashScopeClient:
    global _dashscope_client
    if _dashscope_client is None:
        _dashscope_client = DashScopeClient()
    return _dashscope_client


async def close_dashscope_client() -> None:
    global _dashscope_client
    if _dashscope_client is not None:
        await _dashscope_client.close()
        _dashscope_client = None
