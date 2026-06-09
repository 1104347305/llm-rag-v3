from __future__ import annotations

from time import perf_counter
from typing import Any

from src.main.python.config import Settings, settings
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline
from src.main.python.db.dashscope import DashScopeUnavailable, get_dashscope_client


class AgnoRAGAgent:
    def __init__(self, runtime_settings: Settings = settings) -> None:
        """初始化 Agent。"""
        # pipeline 在 __init__ 中复用单例

        self.settings = runtime_settings
        self._agent: Any | None = None
        self._pipeline = RetrievalPipeline.get()

    def answer(
        self,
        project_id: str,
        query: str,
        retrieval_query: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **context_kwargs: object,
    ) -> dict[str, object]:
        t_start = perf_counter()
        effective_retrieval_query = retrieval_query or build_retrieval_query(query, history or [])
        context = self._pipeline.retrieve(project_id=project_id, query=effective_retrieval_query, **context_kwargs)
        t_retrieval_done = perf_counter()
        prompt = build_user_prompt(query, context, history or [])
        answer, llm_error, engine = self._run_agent(prompt, session_id=session_id, user_id=user_id)
        t_total = perf_counter()
        return {
            "answer": answer,
            "llm_error": llm_error,
            "context": context,
            "session_id": session_id,
            "agent_engine": engine,
            "end_flag": 1,
            "total_ms": round((t_total - t_start) * 1000),
            "retrieval_ms": round((t_retrieval_done - t_start) * 1000),
            "llm_ms": round((t_total - t_retrieval_done) * 1000),
            "rewritten_query": effective_retrieval_query if effective_retrieval_query != query else None,
        }

    def answer_stream(
        self, project_id: str, query: str,
        retrieval_query: str | None = None,
        session_id: str | None = None, user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **context_kwargs: object,
    ):
        """流式问答：检索（同步）→ LLM 流式生成。

        Returns:
            (context_dict, chunk_generator)
            context_dict: 检索结果（pages, metrics 等）
            chunk_generator: yield 文本块的生成器
        """
        effective_retrieval_query = retrieval_query or build_retrieval_query(query, history or [])
        context = self._pipeline.retrieve(project_id=project_id, query=effective_retrieval_query, **context_kwargs)
        if effective_retrieval_query != query:
            context["rewritten_query"] = effective_retrieval_query
        prompt = build_user_prompt(query, context, history or [])

        def _generate():
            try:
                client = get_dashscope_client()
                messages = [{"role": "system", "content": self.settings.system_prompt}, {"role": "user", "content": prompt}]
                yield from client.chat_stream(messages)
            except DashScopeUnavailable:
                yield "[错误: LLM 调用失败]"
            except Exception as exc:
                yield f"[错误: LLM 调用异常: {exc}]"

        return context, _generate()

    def _run_agent(self, prompt: str, session_id: str | None, user_id: str | None) -> tuple[str, str | None, str]:
        """执行 Agent 或回退到 DashScope 直接调用。"""

        if self.settings.enable_agno_agent:
            try:
                response = self._get_agent().run(prompt, session_id=session_id, user_id=user_id)
                return extract_response_content(response), None, "agno"
            except (ImportError, DashScopeUnavailable) as exc:
                fallback_error = str(exc)
            except Exception as exc:
                fallback_error = f"Agno agent unavailable: {exc}"
        else:
            fallback_error = "Agno agent is disabled"

        try:
            answer = get_dashscope_client().chat([{"role": "system", "content": self.settings.system_prompt}, {"role": "user", "content": prompt}])
            return answer, None, "dashscope_fallback"
        except DashScopeUnavailable as exc:
            return "", f"{fallback_error}; fallback failed: {exc}", "unavailable"

    def _get_agent(self) -> Any:
        """延迟创建 Agno Agent 实例。"""

        if self._agent is not None:
            return self._agent
        if not self.settings.dashscope_api_key:
            raise DashScopeUnavailable("DASHSCOPE_API_KEY is not configured")

        try:
            from agno.agent import Agent
            from agno.db.sqlite import SqliteDb
            from agno.models.openai import OpenAIChat
        except ImportError as exc:
            raise ImportError("Agno is not installed; install project dependencies with agno enabled") from exc

        self.settings.agno_session_db_path.parent.mkdir(parents=True, exist_ok=True)
        model_kwargs: dict[str, object] = {
            "id": self.settings.llm_model,
            "api_key": self.settings.dashscope_api_key,
            "base_url": self.settings.dashscope_chat_base_url,
            "role_map": {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"},
        }
        if self.settings.llm_reasoning_effort:
            model_kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
        self._agent = Agent(
            model=OpenAIChat(**model_kwargs),
            db=SqliteDb(db_file=str(self.settings.agno_session_db_path)),
            add_history_to_context=True,
            num_history_runs=self.settings.agno_history_runs,
        )
        return self._agent


def build_retrieval_query(query: str, history: list[dict[str, str]], max_user_messages: int | None = None) -> str:
    """构建检索查询：有历史时用 LLM 改写消除指代，无历史时直接返回。"""
    if not history:
        return query

    from time import perf_counter
    from src.main.python.config import settings
    from src.main.python.utils.logging import get_logger
    _log = get_logger(__name__)

    recent = _recent_exchange(history, turns=2)
    if not recent:
        return query

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
        for m in recent
    )
    rewrite_prompt = settings.query_rewrite_user_template.format(history=history_text, query=query)

    t_start = perf_counter()
    error_detail = ""
    try:
        from src.main.python.db.dashscope import DashScopeUnavailable, get_dashscope_client
        raw = get_dashscope_client().chat(
            [
                {"role": "system", "content": settings.query_rewrite_system_prompt},
                {"role": "user", "content": rewrite_prompt},
            ],
            timeout=settings.rewrite_timeout,
            model=settings.rewrite_model,
            max_tokens=128,
        )
        elapsed = round((perf_counter() - t_start) * 1000)

        import json as _json
        try:
            result = _json.loads(raw.strip().strip("`").strip())
        except _json.JSONDecodeError:
            _log.warning(f"[Query改写] {elapsed}ms | JSON解析失败: {raw!r}")
            return query

        if result.get("rewrite") and result.get("query"):
            rewritten = str(result["query"]).strip()
            if rewritten != query:
                _log.info(f"[Query改写] {elapsed}ms | {rewritten!r}")
                return rewritten

        _log.info(f"[Query改写] {elapsed}ms | 无需改写")
        return query
    except DashScopeUnavailable as exc:
        elapsed = round((perf_counter() - t_start) * 1000)
        _log.warning(f"[Query改写] {elapsed}ms | API不可用: {exc}")
    except Exception as exc:
        elapsed = round((perf_counter() - t_start) * 1000)
        _log.warning(f"[Query改写] {elapsed}ms | 异常: {type(exc).__name__}: {exc}")

    return query


def _recent_exchange(history: list[dict[str, str]], turns: int = 2) -> list[dict[str, str]]:
    """取最近 N 轮对话（每轮 = user + assistant）。"""
    result: list[dict[str, str]] = []
    user_count = 0
    for msg in reversed(history):
        result.insert(0, msg)
        if msg.get("role") == "user":
            user_count += 1
            if user_count >= turns:
                break
    return result


def build_user_prompt(query: str, context: dict[str, object], history: list[dict[str, str]] | None = None) -> str:
    """构建 LLM User Prompt：问题 + 历史 + 上下文 + 指令。"""

    from src.main.python.config import settings

    return "\n".join(
        [
            f"问题：{query}",
            "",
            "## Conversation History",
            format_history(history or []),
            "",
            "## Page List",
            str(context.get("page_list") or "(No pages found)"),
            "",
            "## Index",
            str(context.get("index") or "(No index)"),
            "",
            "## Wiki Pages",
            str(context.get("pages_context") or "(No context)"),
            "",
            settings.user_prompt_inline_instruction,
        ]
    )


def format_history(history: list[dict[str, str]]) -> str:
    """格式化对话历史为文本。"""

    if not history:
        return "(No conversation history)"
    from src.main.python.config import settings
    lines: list[str] = []
    for message in history[-settings.history_normalize_max:]:
        role = message.get("role", "")
        content = message.get("content", "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(No conversation history)"


def extract_response_content(response: Any) -> str:
    """从 Agno Agent 响应中提取文本内容。"""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)
