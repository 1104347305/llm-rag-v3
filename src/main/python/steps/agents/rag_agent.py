from __future__ import annotations

from time import perf_counter
from typing import Any

from loguru import logger
from src.main.python.config import Settings, settings
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline
from src.main.python.services.request_service import (
    DashScopeClient,
    DashScopeUnavailable,
    get_dashscope_client,
)
from src.main.python.utils.concurrency import ServiceOverloaded


class AgnoRAGAgent:
    def __init__(
        self,
        runtime_settings: Settings = settings,
        client: DashScopeClient | None = None,
    ) -> None:
        self.settings = runtime_settings
        self._pipeline = RetrievalPipeline.get()
        self.client = client or get_dashscope_client()
        self.model = self.settings.llm_model
        self.rewrite_model = self.settings.rewrite_model
        self.system_prompt = self.settings.system_prompt
        self.rewrite_system_prompt = self.settings.query_rewrite_system_prompt

    async def build_retrieval_query(self, query: str, history: list[dict[str, str]],
                                     max_user_messages: int | None = None) -> str:
        if not history:
            return query
        recent = self._recent_exchange(history, turns=2)
        if not recent:
            return query
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in recent)
        rewrite_prompt = self.settings.query_rewrite_user_template.format(
            history=history_text, query=query)

        t_start = perf_counter()
        try:
            raw = await self._chat(
                [{"role": "system", "content": self.settings.query_rewrite_system_prompt},
                 {"role": "user", "content": rewrite_prompt}],
                model=self.rewrite_model, max_tokens=128)
            elapsed = round((perf_counter() - t_start) * 1000)

            import json as _json
            try:
                result = _json.loads(raw.strip().strip("`").strip())
            except _json.JSONDecodeError:
                logger.warning(f"[Query改写] {elapsed}ms | JSON解析失败: {raw!r}")
                return query
            if result.get("rewrite") and result.get("query"):
                rewritten = str(result["query"]).strip()
                if rewritten != query:
                    logger.info(f"[Query改写] {elapsed}ms | {rewritten!r}")
                    return rewritten
            logger.info(f"[Query改写] {elapsed}ms | 无需改写")
            return query
        except DashScopeUnavailable as exc:
            logger.warning(f"[Query改写] {round((perf_counter()-t_start)*1000)}ms | API不可用: {exc}")
        except Exception as exc:
            logger.warning(f"[Query改写] {round((perf_counter()-t_start)*1000)}ms | 异常: {type(exc).__name__}: {exc}")
        return query

    @staticmethod
    def _recent_exchange(history: list[dict[str, str]], turns: int = 2) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        user_count = 0
        for msg in reversed(history):
            result.insert(0, msg)
            if msg.get("role") == "user":
                user_count += 1
                if user_count >= turns:
                    break
        return result

    def build_user_prompt(self, query: str, context: dict[str, object],
                          history: list[dict[str, str]] | None = None) -> str:
        return "\n".join([
            f"问题：{query}", "",
            "## Conversation History",
            self.format_history(history or []), "",
            "## Page List",
            str(context.get("page_list") or "(No pages found)"), "",
            "## Index",
            str(context.get("index") or "(No index)"), "",
            "## Wiki Pages",
            str(context.get("pages_context") or "(No context)"), "",
            self.settings.user_prompt_inline_instruction,
        ])

    @staticmethod
    def format_history(history: list[dict[str, str]]) -> str:
        if not history:
            return "(No conversation history)"
        lines = []
        for message in history[-settings.history_normalize_max:]:
            role = message.get("role", "")
            content = message.get("content", "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines) if lines else "(No conversation history)"

    @staticmethod
    def extract_response_content(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    async def answer(self, project_id: str, query: str,
                     retrieval_query: str | None = None,
                     session_id: str | None = None, user_id: str | None = None,
                     history: list[dict[str, str]] | None = None,
                     **context_kwargs: object) -> dict[str, object]:
        t_start = perf_counter()
        effective_retrieval_query = retrieval_query or await self.build_retrieval_query(query, history or [])
        context = await self._pipeline.retrieve(project_id=project_id, query=effective_retrieval_query, **context_kwargs)
        t_retrieval_done = perf_counter()
        prompt = self.build_user_prompt(query, context, history or [])
        answer, llm_error, engine = await self._run_agent(prompt, session_id=session_id, user_id=user_id)
        t_total = perf_counter()
        return {
            "answer": answer, "llm_error": llm_error, "context": context,
            "session_id": session_id, "agent_engine": engine, "end_flag": 1,
            "total_ms": round((t_total - t_start) * 1000),
            "retrieval_ms": round((t_retrieval_done - t_start) * 1000),
            "llm_ms": round((t_total - t_retrieval_done) * 1000),
            "rewritten_query": effective_retrieval_query if effective_retrieval_query != query else None,
        }

    async def answer_stream(self, project_id: str, query: str,
                            retrieval_query: str | None = None,
                            session_id: str | None = None, user_id: str | None = None,
                            history: list[dict[str, str]] | None = None,
                            **context_kwargs: object):
        effective_retrieval_query = retrieval_query or await self.build_retrieval_query(query, history or [])
        context = await self._pipeline.retrieve(project_id=project_id, query=effective_retrieval_query, **context_kwargs)
        if effective_retrieval_query != query:
            context["rewritten_query"] = effective_retrieval_query
        prompt = self.build_user_prompt(query, context, history or [])

        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
        stream = await self.client.create_chat_stream(messages, model=self.model)

        async def _generate_opened_stream():
            try:
                async for chunk in stream:
                    yield chunk
            except ServiceOverloaded:
                raise
            except Exception as exc:
                yield f"[错误: LLM 调用失败: {exc}]"

        return context, _generate_opened_stream()

    async def _run_agent(self, prompt: str, session_id: str | None, user_id: str | None) -> tuple[str, str | None, str]:
        try:
            answer = await self._chat(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": prompt}])
            return answer, None, "async_openai"
        except ServiceOverloaded:
            raise
        except Exception as exc:
            return "", str(exc), "unavailable"

    async def _chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            **self._reasoning_kwargs(),
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return await self.client.chat(
            messages,
            model=str(body["model"]),
            max_tokens=max_tokens,
        )

    def _reasoning_kwargs(self) -> dict[str, str]:
        if not self.settings.llm_reasoning_effort:
            return {}
        return {"reasoning_effort": self.settings.llm_reasoning_effort}
