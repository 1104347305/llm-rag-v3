"""RAG 系统 API 路由——所有端点集中定义。

匹配 llm_client_search 框架模式：所有路由在一个文件中管理。
"""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.main.python.config import settings
from src.main.python.models.schemas import (
    AnswerRequest, ChatRequest, ChatResponse, ChatData, ChatStatus, ChatExtraOutput,
    ContextRequest, DataProcessRequest, IndexRequest,
)
from src.main.python.services import get_rag_service
from src.main.python.utils.logging import get_logger, log_event

logger = get_logger(__name__)

router = APIRouter(tags=["rag"])
_timeout_executor = ThreadPoolExecutor(max_workers=4)


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _run_with_timeout(fn) -> dict[str, object]:
    """在线程池中执行同步任务，超时后抛 504。"""
    try:
        return _timeout_executor.submit(fn).result(timeout=settings.sync_timeout)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FutureTimeout:
        raise HTTPException(status_code=504, detail=f"request timed out after {settings.sync_timeout}s")


# ═══════════════════════════════════════════════════════════
# 检索端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/context")
def context_endpoint(request: ContextRequest) -> dict[str, object]:
    """仅检索上下文，不调用 LLM。返回相关页面及分块内容。"""
    return _run_with_timeout(lambda: get_rag_service().retrieve_context(**request.model_dump()))


@router.post("/rag/search/debug")
def debug_endpoint(request: ContextRequest) -> dict[str, object]:
    """检索调试端点，返回额外 debug 字段（各召回原始结果、RRF 融合详情）。"""
    payload = request.model_dump()
    payload["debug"] = True
    return _run_with_timeout(lambda: get_rag_service().retrieve_context(**payload))


# ═══════════════════════════════════════════════════════════
# 问答端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/answer")
def answer_endpoint(request: AnswerRequest) -> dict[str, object]:
    """同步问答：等待完整答案后返回。"""
    return _run_with_timeout(lambda: get_rag_service().answer_query(**request.model_dump()))


@router.post("/rag/answer/stream")
def answer_stream_endpoint(request: AnswerRequest):
    """流式问答：SSE 格式。

    每个事件结构与 /rag/answer 同步响应一致，通过 end_flag 区分状态:
      0 = 未结束（answer 为当前累加内容）
      1 = 已结束（answer 为完整答案）
    """
    t_start = perf_counter()
    try:
        context, generator = get_rag_service().answer_query_stream(**request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    t_retrieval_done = perf_counter()
    retrieval_ms = round((t_retrieval_done - t_start) * 1000)
    session_id = request.session_id

    def _build_event(answer: str, end_flag: int, llm_start: float, now: float) -> dict[str, object]:
        """构建单个 SSE 事件 JSON。"""
        return {
            "answer": answer,
            "end_flag": end_flag,
            "total_ms": round((now - t_start) * 1000),
            "retrieval_ms": retrieval_ms,
            "llm_ms": round((now - llm_start) * 1000),
            "llm_error": None,
            "context": {
                "pages": context.get("pages", []),
                "metrics": context.get("metrics", {}),
                "fallback_reasons": context.get("fallback_reasons", []),
                "rewritten_query": context.get("rewritten_query"),
            },
            "session_id": session_id,
            "agent_engine": "dashscope",
        }

    def _sse():
        """SSE 生成器：逐块推送累加的 answer 文本。"""
        print("[SSE] 开始流式生成...", flush=True)
        first = True
        full_answer = ""
        char_count = 0
        log_interval = settings.stream_log_interval
        next_log_at = log_interval
        t_llm_start = perf_counter()

        try:
            for chunk in generator:
                if not chunk:
                    continue
                if first:
                    print(f"[SSE] 首个 chunk: {chunk[:50]}...", flush=True)
                    first = False

                full_answer += chunk
                char_count += len(chunk)
                while char_count >= next_log_at:
                    print(f"\n[流式输出] {next_log_at} 字:\n{full_answer}\n", flush=True)
                    log_event(logger, 20, "stream.progress", f"流式输出 {next_log_at} 字", chars=next_log_at)
                    next_log_at += log_interval

                now = perf_counter()
                yield f"data: {json.dumps(_build_event(chunk, 0, t_llm_start, now), ensure_ascii=False)}\n\n"

            if first:
                print("[SSE] 生成器未产生任何内容", flush=True)
            else:
                # 打印最后一轮未到日志间隔的剩余内容
                print(f"\n[流式输出] 完整 {char_count} 字:\n{full_answer}\n", flush=True)
                log_event(logger, 20, "stream.progress", f"流式完成 {char_count} 字", chars=char_count)

            now = perf_counter()
            print(f"[SSE] 流式完成，共 {char_count} 字", flush=True)
            yield f"data: {json.dumps(_build_event(full_answer, 1, t_llm_start, now), ensure_ascii=False)}\n\n"

        except Exception as exc:
            print(f"[SSE] 流式异常: {exc}", flush=True)
            raise

    return StreamingResponse(_sse(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════
# 索引端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/index/project")
def index_project_endpoint(request: IndexRequest) -> dict[str, object]:
    """异步索引项目：扫描 Markdown → 解析 → 分块 → 嵌入 → 三层存储写入。"""
    return get_rag_service().run_index_job(
        request.project_id, Path(request.project_path), request.force, request.build_embeddings)


@router.post("/data/process")
def process_data_endpoint(request: DataProcessRequest) -> dict[str, object]:
    """数据处理快捷入口，默认 project_id='pingan-zhenxiang'。"""
    return get_rag_service().run_index_job(
        request.project_id, Path(request.data_path), request.force, request.build_embeddings)


# ═══════════════════════════════════════════════════════════
# 会话端点
# ═══════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}")
def session_endpoint(session_id: str) -> dict[str, object]:
    """获取会话历史。"""
    return get_rag_service().get_session(session_id)


@router.delete("/sessions/{session_id}")
def clear_session_endpoint(session_id: str) -> dict[str, object]:
    """清除会话历史。"""
    return get_rag_service().clear_session(session_id)


# ═══════════════════════════════════════════════════════════
# 健康检查端点
# ═══════════════════════════════════════════════════════════

@router.get("/health")
def health_endpoint() -> dict[str, object]:
    """健康检查：返回服务配置和组件状态。"""
    return {
        "status": "ok",
        "storage_dir": str(settings.storage_dir),
        "elasticsearch": {
            "configured": settings.es_enabled,
            "indexing_enabled": settings.es_indexing_enabled,
            "retrieval_enabled": settings.es_retrieval_enabled,
            "index_prefix": settings.es_index_prefix,
        },
        "retrieval": {
            "vector_enabled": settings.enable_vector_retrieval,
            "pgvector_enabled": settings.pgvector_enabled,
            "local_lexical_enabled": settings.enable_local_lexical_retrieval,
            "local_lexical_max_chunks": settings.local_lexical_max_chunks,
            "local_vector_max_chunks": settings.local_vector_max_chunks,
            "default_max_context_size": settings.default_max_context_size,
            "default_top_pages": settings.default_top_pages,
            "default_bm25_top_k": settings.default_bm25_top_k,
            "default_vector_top_k": settings.default_vector_top_k,
            "default_rerank_top_k": settings.default_rerank_top_k,
        },
        "indexing": {"build_embeddings": settings.build_embeddings},
        "logging": {"level": settings.log_level, "file": settings.log_file},
        "sqlite": {"fts5_available": _sqlite_fts5_available()},
        "dashscope": {
            "configured": settings.dashscope_enabled,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "llm_model": settings.llm_model,
        },
    }


@router.get("/health/ready")
def health_ready_endpoint() -> dict[str, object]:
    """运行时深度就绪检查，返回各组件状态。"""
    checks: dict[str, dict[str, Any]] = {}
    service = get_rag_service()

    try:
        from src.main.python.db.local_store import LocalStore
        store = LocalStore.get()
        projects = [
            p.stem for p in store.index_dir.glob("*.json")
            if store.has_sqlite_index(p.stem)
        ]
        checks["indexes"] = {"ready": len(projects) > 0, "count": len(projects), "projects": projects[:10]}
    except Exception as exc:
        checks["indexes"] = {"ready": False, "error": str(exc)}

    checks["sessions"] = {"active": len(service._session_history), "ready": True}
    all_ready = all(c.get("ready", True) for c in checks.values())
    return {"status": "ready" if all_ready else "degraded", "components": checks}


def _sqlite_fts5_available() -> bool:
    """检查 SQLite 编译时是否包含 FTS5 扩展。"""
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.execute("create virtual table fts_check using fts5(content)")
        return True
    except sqlite3.Error:
        return False


# ═══════════════════════════════════════════════════════════
# AskBob 协议端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/chat")
def chat_endpoint(request: ChatRequest) -> dict[str, object]:
    """AskBob 协议同步问答接口。

    入参映射: user_text→query, source→project_id, session_id→session_id
    出参映射: answer→data.robot_text, pages→data.extra_output_params.pages
    """
    t_start = perf_counter()
    result = get_rag_service().answer_query(
        project_id=request.source,
        query=request.user_text,
        session_id=request.session_id or None,
        user_id=request.user_id or None,
        history=[],
    )
    total_ms = round((perf_counter() - t_start) * 1000)
    context = result.get("context", {})
    return {
        "code": 0,
        "data": {
            "robot_text": str(result.get("answer") or ""),
            "source": request.source,
            "end_flag": 1,
            "status": {"report_ready_flag": 0, "return_task_flag": 1},
            "extra_output_params": {
                "query": request.user_text,
                "rewritten_query": context.get("rewritten_query", ""),
                "first_frame_time": context.get("metrics", {}).get("bm25_latency_ms", 0),
                "final_frame_time": total_ms,
                "pages": context.get("pages", []),
                "metrics": context.get("metrics", {}),
            },
        },
    }


@router.post("/rag/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    """AskBob 协议流式问答接口（SSE 格式）。

    入参映射同 /rag/chat，出参每个 SSE 事件为 ChatResponse JSON。
    end_flag: 0=未结束（robot_text 为增量文本），1=已结束（robot_text 为完整答案）。
    """
    t_start = perf_counter()
    try:
        context, generator = get_rag_service().answer_query_stream(
            project_id=request.source,
            query=request.user_text,
            session_id=request.session_id or None,
            user_id=request.user_id or None,
            history=[],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    t_retrieval_done = perf_counter()
    retrieval_ms = round((t_retrieval_done - t_start) * 1000)

    rewritten_query = context.get("rewritten_query", "")

    def _build_event(answer: str, end_flag: int, llm_start: float, now: float) -> dict[str, object]:
        return {
            "code": 0,
            "data": {
                "robot_text": answer,
                "source": request.source,
                "end_flag": end_flag,
                "status": {"report_ready_flag": 0, "return_task_flag": 1},
                "extra_output_params": {
                    "query": request.user_text,
                    "rewritten_query": rewritten_query,
                    "first_frame_time": round((now - t_start) * 1000) if end_flag == 0 else retrieval_ms,
                    "final_frame_time": round((now - t_start) * 1000),
                    "pages": context.get("pages", []),
                    "metrics": context.get("metrics", {}),
                },
            },
        }

    def _sse():
        print("[SSE] AskBob 流式开始...", flush=True)
        full_answer = ""
        char_count = 0
        log_interval = settings.stream_log_interval
        next_log_at = log_interval
        t_llm_start = perf_counter()

        try:
            for chunk in generator:
                if not chunk:
                    continue
                full_answer += chunk
                char_count += len(chunk)
                while char_count >= next_log_at:
                    print(f"\n[流式输出] {next_log_at} 字:\n{full_answer}\n", flush=True)
                    log_event(logger, 20, "stream.progress", f"流式输出 {next_log_at} 字", chars=next_log_at)
                    next_log_at += log_interval

                now = perf_counter()
                yield f"data: {json.dumps(_build_event(chunk, 0, t_llm_start, now), ensure_ascii=False)}\n\n"

            if char_count > 0:
                print(f"\n[流式输出] 完整 {char_count} 字:\n{full_answer}\n", flush=True)
                log_event(logger, 20, "stream.progress", f"流式完成 {char_count} 字", chars=char_count)

            now = perf_counter()
            print(f"[SSE] AskBob 流式完成，共 {char_count} 字", flush=True)
            yield f"data: {json.dumps(_build_event(full_answer, 1, t_llm_start, now), ensure_ascii=False)}\n\n"

        except Exception as exc:
            print(f"[SSE] 流式异常: {exc}", flush=True)
            raise

    return StreamingResponse(_sse(), media_type="text/event-stream")
