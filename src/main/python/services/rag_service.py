from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

from src.main.python.config import Settings, settings
from src.main.python.steps.indexing.worker import index_project
from src.main.python.models.domain import JobStatus
from src.main.python.steps.retrieval.answer import answer_query
from src.main.python.steps.agents.rag_agent import AgnoRAGAgent, build_retrieval_query
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline
from src.main.python.utils.logging import get_logger, log_event

# 会话 / 作业的过期时间（秒）
SESSION_TTL = 3600   # 1 小时
JOB_TTL = 86400      # 24 小时
MAX_SESSIONS = 10000  # 会话上限
MAX_JOBS = 1000       # 作业上限


class RAGService:
    def __init__(self, runtime_settings: Settings = settings) -> None:
        """初始化 RAG 服务。"""
        self.settings = runtime_settings
        self._jobs: dict[str, JobStatus] = {}
        self._session_history: dict[str, list[dict[str, str]]] = {}
        self._session_times: dict[str, float] = {}
        self._job_times: dict[str, float] = {}
        self._logger = get_logger(__name__)
        self._pipeline = RetrievalPipeline.get()
        self._lock = threading.Lock()

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

    def index_project(self, project_id: str, project_path: Path, force: bool = False, build_embeddings: bool | None = None) -> dict[str, object]:
        should_build_embeddings = self.settings.build_embeddings if build_embeddings is None else build_embeddings
        return index_project(project_id, project_path, force=force, build_embeddings=should_build_embeddings)

    def run_index_job(self, project_id: str, project_path: Path, force: bool = False, build_embeddings: bool | None = None) -> dict[str, object]:
        should_build_embeddings = self.settings.build_embeddings if build_embeddings is None else build_embeddings
        job_id = f"job_{uuid4().hex[:12]}"
        status = JobStatus(job_id=job_id, status="running")
        with self._lock:
            self._purge_expired()
            self._jobs[job_id] = status
            self._job_times[job_id] = time()
        start = perf_counter()
        log_event(
            self._logger,
            20,
            "job.index.start",
            "index job started",
            job_id=job_id,
            project_id=project_id,
            project_path=str(project_path),
            force=force,
            build_embeddings=should_build_embeddings,
        )
        try:
            result = self.index_project(project_id, project_path, force=force, build_embeddings=should_build_embeddings)
            status.status = "completed"
            status.pages_total = int(result["pages_indexed"])
            status.pages_done = int(result["pages_indexed"])
            status.chunks_indexed = int(result["chunks_indexed"])
            status.embeddings_done = int(result["chunks_indexed"]) if should_build_embeddings else 0
            payload = {"job_id": job_id, "status": status.status, **result}
            log_event(self._logger, 20, "job.index.completed", "index job completed", **payload)
            return payload
        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            log_event(self._logger, 40, "job.index.failed", "index job failed", job_id=job_id, project_id=project_id, error=str(exc))
            raise
        finally:
            status.duration_ms = round((perf_counter() - start) * 1000, 3)

    def get_job(self, job_id: str) -> dict[str, object] | None:
        status = self._jobs.get(job_id)
        return asdict(status) if status else None

    def retrieve_context(self, project_id: str, query: str, **kwargs: object) -> dict[str, object]:
        return self._pipeline.retrieve(project_id=project_id, query=query, **kwargs)

    def answer_query(
        self,
        project_id: str,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        stored_history = self._session_history.get(session_id, []) if session_id else []
        effective_history = normalize_history(history if history else stored_history)
        retrieval_query = build_retrieval_query(query, effective_history)
        result = answer_query(
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
            with self._lock:
                self._purge_expired()
                self._session_history[session_id] = normalize_history(updated_history, max_messages=self.settings.session_max_messages)
                self._session_times[session_id] = time()
                result["history"] = self._session_history[session_id]
        return result

    def answer_query_stream(
        self, project_id: str, query: str,
        session_id: str | None = None, user_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: object,
    ):
        """流式问答：检索 → 返回 (context, stream_generator)。"""
        stored_history = self._session_history.get(session_id, []) if session_id else []
        effective_history = normalize_history(history if history else stored_history)
        self._logger.info(
            "[会话状态] session_id={} | history请求传入={}条 | 服务端存储={}条 | 有效history={}条",
            session_id,
            len(history) if history else 0,
            len(stored_history),
            len(effective_history),
        )
        retrieval_query = build_retrieval_query(query, effective_history)
        context, generator = AgnoRAGAgent(self.settings).answer_stream(
            project_id=project_id, query=query,
            retrieval_query=retrieval_query,
            session_id=session_id, user_id=user_id,
            history=effective_history, **kwargs,
        )

        # 更新会话历史
        if session_id:
            collected: list[str] = []
            original_gen = generator

            def _tracking_gen():
                for chunk in original_gen:
                    collected.append(chunk)
                    yield chunk
                answer = "".join(collected)
                updated = [*effective_history, {"role": "user", "content": query}]
                if answer:
                    updated.append({"role": "assistant", "content": answer})
                with self._lock:
                    self._purge_expired()
                    self._session_history[session_id] = normalize_history(updated, max_messages=self.settings.session_max_messages)
                    self._session_times[session_id] = time()
                self._logger.info(
                    "[会话保存] session_id={} | 保存了{}条历史（含当前轮）",
                    session_id, len(self._session_history.get(session_id, [])),
                )

            return context, _tracking_gen()

        return context, generator

    def get_session(self, session_id: str) -> dict[str, object]:
        return {"session_id": session_id, "history": self._session_history.get(session_id, [])}

    def clear_session(self, session_id: str) -> dict[str, object]:
        self._session_history.pop(session_id, None)
        return {"session_id": session_id, "cleared": True}


rag_service = RAGService()


def get_rag_service() -> RAGService:
    """获取全局 RAGService 单例。"""

    return rag_service


def normalize_history(history: list[dict[str, str]], max_messages: int | None = None) -> list[dict[str, str]]:
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
