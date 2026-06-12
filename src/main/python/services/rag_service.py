from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from src.main.python.config import Settings, settings
from src.main.python.steps.indexing.worker import IndexBuilder
from src.main.python.models.domain import JobStatus
from src.main.python.steps.agents.rag_agent import AgnoRAGAgent
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline
from loguru import logger

# 会话 / 作业的过期时间（秒）
SESSION_TTL = 3600   # 1 小时
JOB_TTL = 86400      # 24 小时
MAX_SESSIONS = 10000  # 会话上限
MAX_JOBS = 1000       # 作业上限


class ProjectIndexBusy(RuntimeError):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"indexing is already running for project: {project_id}")


class RAGService:
    def __init__(self,
                 runtime_settings: Settings = settings,
                 builder: IndexBuilder | None = None,
                 pipeline: RetrievalPipeline | None = None,
                 agent: AgnoRAGAgent | None = None) -> None:
        self.settings = runtime_settings
        self._builder = builder or IndexBuilder()
        self._pipeline = pipeline or RetrievalPipeline.get()
        self._agent = agent or AgnoRAGAgent(self.settings)
        self._jobs: dict[str, JobStatus] = {}
        self._session_history: dict[str, list[dict[str, str]]] = {}
        self._session_times: dict[str, float] = {}
        self._job_times: dict[str, float] = {}
        self._indexing_projects: set[str] = set()
        self._lock = asyncio.Lock()

    def _purge_expired(self) -> None:
        """清理过期会话和作业（内部调用，需在锁内执行）。"""
        now = time()
        for sid in [s for s, t in self._session_times.items() if now - t > SESSION_TTL]:
            self._session_history.pop(sid, None)
            self._session_times.pop(sid, None)
        for jid in [j for j, t in self._job_times.items() if now - t > JOB_TTL]:
            self._jobs.pop(jid, None)
            self._job_times.pop(jid, None)
        # 超出上限时淘汰最旧的
        while len(self._session_history) > MAX_SESSIONS:
            oldest = min(self._session_times, key=self._session_times.get)
            self._session_history.pop(oldest, None)
            self._session_times.pop(oldest, None)
        while len(self._jobs) > MAX_JOBS:
            oldest = min(self._job_times, key=self._job_times.get)
            self._jobs.pop(oldest, None)
            self._job_times.pop(oldest, None)

    async def index_project(self, project_id: str, project_path: Path, force: bool = False, build_embeddings: bool | None = None) -> dict[str, object]:
        should_build_embeddings = self.settings.build_embeddings if build_embeddings is None else build_embeddings
        return await self._builder.build(project_id, project_path, force=force, build_embeddings=should_build_embeddings)

    async def run_index_job(self, project_id: str, project_path: Path, force: bool = False, build_embeddings: bool | None = None) -> dict[str, object]:
        should_build_embeddings = self.settings.build_embeddings if build_embeddings is None else build_embeddings
        job_id = f"job_{uuid4().hex[:12]}"
        status = JobStatus(job_id=job_id, status="running")
        async with self._lock:
            self._purge_expired()
            if project_id in self._indexing_projects:
                raise ProjectIndexBusy(project_id)
            self._indexing_projects.add(project_id)
            self._jobs[job_id] = status
            self._job_times[job_id] = time()
        start = perf_counter()
        logger.bind(event="job.index.start").info(
            "index job started",
            job_id=job_id, project_id=project_id,
            project_path=str(project_path),
            force=force, build_embeddings=should_build_embeddings)
        try:
            result = await self.index_project(project_id, project_path, force=force, build_embeddings=should_build_embeddings)
            status.status = "completed"
            status.pages_total = int(result["pages_indexed"])
            status.pages_done = int(result["pages_indexed"])
            status.chunks_indexed = int(result["chunks_indexed"])
            status.embeddings_done = int(result["chunks_indexed"]) if should_build_embeddings else 0
            payload = {"job_id": job_id, "status": status.status, **result}
            logger.bind(event="job.index.completed").info("index job completed", **payload)
            return payload
        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            logger.bind(event="job.index.failed").error("index job failed", job_id=job_id, project_id=project_id, error=str(exc))
            raise
        finally:
            status.duration_ms = round((perf_counter() - start) * 1000, 3)
            async with self._lock:
                self._indexing_projects.discard(project_id)

    async def retrieve_context(self, project_id: str, query: str, **kwargs: object) -> dict[str, object]:
        return await self._pipeline.retrieve(project_id=project_id, query=query, **kwargs)

    async def answer_query(
        self,
        project_id: str,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        stored_history = self._session_history.get(session_id, []) if session_id else []
        effective_history = self.normalize_history(history if history else stored_history)
        retrieval_query = await self._agent.build_retrieval_query(query, effective_history)
        result = await self._agent.answer(
            project_id=project_id,
            query=query,
            session_id=session_id,
            user_id=user_id,
            history=effective_history,
            retrieval_query=retrieval_query,
            **kwargs,
        )
        if session_id:
            updated_history = [*effective_history, {"role": "user", "content": query}]
            answer = str(result.get("answer") or "")
            if answer:
                updated_history.append({"role": "assistant", "content": answer})
            async with self._lock:
                self._purge_expired()
                self._session_history[session_id] = self.normalize_history(
                    updated_history, max_messages=self.settings.session_max_messages
                )
                self._session_times[session_id] = time()
                result["history"] = self._session_history[session_id]
        return result

    async def answer_query_stream(
        self, project_id: str, query: str,
        session_id: str | None = None, user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: object,
    ):
        """流式问答：检索 → 返回 (context, stream_generator)。"""
        stored_history = self._session_history.get(session_id, []) if session_id else []
        effective_history = self.normalize_history(history if history else stored_history)
        logger.info(
            "[会话状态] session_id={} | history请求传入={}条 | 服务端存储={}条 | 有效history={}条",
            session_id,
            len(history) if history else 0,
            len(stored_history),
            len(effective_history),
        )
        retrieval_query = await self._agent.build_retrieval_query(query, effective_history)
        context, generator = await self._agent.answer_stream(
            project_id=project_id, query=query,
            retrieval_query=retrieval_query,
            session_id=session_id, user_id=user_id,
            history=effective_history, **kwargs,
        )

        # 更新会话历史
        if session_id:
            collected: list[str] = []

            async def _tracking_gen():
                async for chunk in generator:
                    collected.append(chunk)
                    yield chunk
                answer = "".join(collected)
                updated = [*effective_history, {"role": "user", "content": query}]
                if answer:
                    updated.append({"role": "assistant", "content": answer})
                async with self._lock:
                    self._purge_expired()
                    self._session_history[session_id] = self.normalize_history(
                        updated, max_messages=self.settings.session_max_messages
                    )
                    self._session_times[session_id] = time()
                logger.info(
                    "[会话保存] session_id={} | 保存了{}条历史（含当前轮）",
                    session_id, len(self._session_history.get(session_id, [])),
                )

            return context, _tracking_gen()

        return context, generator

    async def get_session(self, session_id: str) -> dict[str, object]:
        async with self._lock:
            self._purge_expired()
            return {"session_id": session_id, "history": self._session_history.get(session_id, [])}

    async def clear_session(self, session_id: str) -> dict[str, object]:
        async with self._lock:
            self._session_history.pop(session_id, None)
            self._session_times.pop(session_id, None)
        return {"session_id": session_id, "cleared": True}

    def normalize_history(self, history: list[dict[str, str]], max_messages: int | None = None) -> list[dict[str, str]]:
        """规范化对话历史：去无效消息、截断到 max_messages 条。"""

        if max_messages is None:
            from src.main.python.config import settings
            max_messages = settings.history_normalize_max
        normalized: list[dict[str, str]] = []
        for message in history[-max_messages:]:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
        return normalized


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
