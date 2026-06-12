"""RAG 系统 API 路由——所有端点集中定义。

匹配 llm_client_search 框架模式：所有路由在一个文件中管理。
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
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
from src.main.python.services.rag_service import RAGService
from loguru import logger


router = APIRouter()


ragService = RAGService(settings)

# ═══════════════════════════════════════════════════════════
# 检索端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/context")
async def context_endpoint(request: ContextRequest) -> dict[str, object]:
    """仅检索上下文，不调用 LLM。返回相关页面及分块内容。"""
    return await ragService.retrieve_context(**request.model_dump())


@router.post("/rag/search/debug")
async def debug_endpoint(request: ContextRequest) -> dict[str, object]:
    """检索调试端点，返回额外 debug 字段（各召回原始结果、RRF 融合详情）。"""
    payload = request.model_dump()
    payload["debug"] = True
    return await ragService.retrieve_context(**payload)

# ═══════════════════════════════════════════════════════════
# 索引端点
# ═══════════════════════════════════════════════════════════

@router.post("/rag/index/project")
async def index_project_endpoint(request: IndexRequest) -> dict[str, object]:
    """异步索引项目：扫描 Markdown → 解析 → 分块 → 嵌入 → 三层存储写入。"""
    return await ragService.run_index_job(
        request.project_id, Path(request.project_path), request.force, request.build_embeddings)


@router.post("/data/process")
async def process_data_endpoint(request: DataProcessRequest) -> dict[str, object]:
    """数据处理快捷入口，默认 project_id='pingan-zhenxiang'。"""
    return await ragService.run_index_job(
        request.project_id, Path(request.data_path), request.force, request.build_embeddings)


# ═══════════════════════════════════════════════════════════
# 会话端点
# ═══════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}")
async def session_endpoint(session_id: str) -> dict[str, object]:
    """获取会话历史。"""
    return await ragService.get_session(session_id)


@router.delete("/sessions/{session_id}")
async def clear_session_endpoint(session_id: str) -> dict[str, object]:
    """清除会话历史。"""
    return await ragService.clear_session(session_id)


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
async def chat_endpoint(request: ChatRequest) -> dict[str, object]:
    """AskBob 协议同步问答接口。

    入参映射: user_text→query, source→project_id, session_id→session_id
    出参映射: answer→data.robot_text, pages→data.extra_output_params.pages
    """
    t_start = perf_counter()
    result = await ragService.answer_query(
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
async def chat_stream_endpoint(request: ChatRequest):
    """AskBob 协议流式问答接口（SSE 格式）。

    入参映射同 /rag/chat，出参每个 SSE 事件为 ChatResponse JSON。
    end_flag: 0=未结束（robot_text 为增量文本），1=已结束（robot_text 为完整答案）。
    """
    t_start = perf_counter()
    try:
        context, generator = await ragService.answer_query_stream(
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

    async def _sse():
        print("[SSE] AskBob 流式开始...", flush=True)
        full_answer = ""
        char_count = 0
        log_interval = settings.stream_log_interval
        next_log_at = log_interval
        t_llm_start = perf_counter()

        try:
            async for chunk in generator:
                if not chunk:
                    continue
                full_answer += chunk
                char_count += len(chunk)
                while char_count >= next_log_at:
                    print(f"\n[流式输出] {next_log_at} 字:\n{full_answer}\n", flush=True)
                    logger.bind(event="stream.progress").info(f"流式输出 {next_log_at} 字", chars=next_log_at)
                    next_log_at += log_interval

                now = perf_counter()
                yield f"data: {json.dumps(_build_event(chunk, 0, t_llm_start, now), ensure_ascii=False)}\n\n"

            if char_count > 0:
                print(f"\n[流式输出] 完整 {char_count} 字:\n{full_answer}\n", flush=True)
                logger.bind(event="stream.progress").info(f"流式完成 {char_count} 字", chars=char_count)

            now = perf_counter()
            print(f"[SSE] AskBob 流式完成，共 {char_count} 字", flush=True)
            yield f"data: {json.dumps(_build_event(full_answer, 1, t_llm_start, now), ensure_ascii=False)}\n\n"

        except Exception as exc:
            print(f"[SSE] 流式异常: {exc}", flush=True)
            raise

    return StreamingResponse(_sse(), media_type="text/event-stream")
