"""
FastAPI 应用主入口
"""
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (str(PACKAGE_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.main.python.api.routes import router
from src.main.python.config import reload_settings, settings
from src.main.python.services import get_rag_service
from src.main.python.utils.logging import configure_logging, get_logger

_log = get_logger(__name__)

# 配置日志
configure_logging()
_log.info("Starting LLM RAG V3 | env={} config={}", os.getenv("ENV", "dev"), os.getenv("RAG_CONFIG_DIR", "."))

# ── Lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("service fully started | host={} port={}", settings.host, settings.port)
    yield
    # 优雅关闭：清理连接池和线程池
    _log.info("service shutting down — cleaning up resources")
    try:
        from src.main.python.db.pgvector_store import PgVectorStore
        PgVectorStore.close()
    except Exception as exc:
        _log.warning("pgvector pool cleanup failed: {}", exc)
    _log.info("service shut down complete")


# 创建 FastAPI 应用
app = FastAPI(
    title="LLM RAG V3",
    description="本地优先的 RAG 系统 — 三路混合检索 + Agno Agent 编排",
    version="0.2.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求计时中间件
@app.middleware("http")
async def _log_request_timing(request: Request, call_next: Callable) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - started) * 1000)
    _log.info("[{}] {} {}ms", response.status_code, request.url.path, elapsed)
    return response

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "LLM RAG V3",
        "version": "0.2.0",
        "status": "running",
    }


@app.post("/admin/reload")
async def admin_reload():
    """热重载 YAML 配置，无需重启。"""
    try:
        result = reload_settings()
        _log.info("配置热重载成功 | env={} changed={}", result["env"], len(result["changed_keys"]))
        return {"status": "ok", **result}
    except Exception as exc:
        _log.error("配置热重载失败: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/jobs/{job_id}")
async def job_endpoint(job_id: str):
    job = get_rag_service().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


if __name__ == "__main__":
    import uvicorn

    _log.info("Starting server on {}:{}", settings.host, settings.port)
    uvicorn.run(
        "src.main.python.main:app",
        host=settings.host,
        port=settings.port,
    )
