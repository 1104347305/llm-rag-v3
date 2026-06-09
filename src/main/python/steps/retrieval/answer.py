from __future__ import annotations

from src.main.python.steps.agents.rag_agent import AgnoRAGAgent
from src.main.python.utils.logging import get_logger, log_event
logger = get_logger(__name__)


def answer_query(
    project_id: str,
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    retrieval_query: str | None = None,
    **context_kwargs,
) -> dict[str, object]:
    """检索 + LLM 问答入口：记录日志后委托给 AgnoRAGAgent。"""
    log_event(
        logger,
        20,
        "answer.start",
        "answer query started",
        project_id=project_id,
        query_length=len(query),
        session_id=session_id,
        user_id=user_id,
        history_turn_count=len(history or []),
        context_kwargs=context_kwargs,
    )
    result = AgnoRAGAgent().answer(
        project_id=project_id,
        query=query,
        retrieval_query=retrieval_query,
        session_id=session_id,
        user_id=user_id,
        history=history,
        **context_kwargs,
    )
    context = result.get("context", {})
    log_event(
        logger,
        20,
        "answer.completed",
        "answer query completed",
        project_id=project_id,
        selected_page_count=len(context.get("pages", [])),
        answer_chars=len(str(result.get("answer") or "")),
        llm_error=result.get("llm_error"),
        session_id=session_id,
        agent_engine=result.get("agent_engine"),
        total_ms=result.get("total_ms"),
        retrieval_ms=result.get("retrieval_ms"),
        llm_ms=result.get("llm_ms"),
        retrieval_metrics=context.get("metrics", {}),
        fallback_reasons=context.get("fallback_reasons", []),
    )
    return result
