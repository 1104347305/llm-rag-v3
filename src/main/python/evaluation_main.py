from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.main.python.api.evaluation import router as evaluation_router
from src.main.python.eval.qa_evaluation import minimax_config
from src.main.python.utils.logging import configure_logging


def create_evaluation_app() -> FastAPI:
    """QA 评估系统 FastAPI app 工厂函数。"""
    configure_logging()

    app = FastAPI(title="QA Evaluation System", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(evaluation_router)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "QA Evaluation System", "version": "0.1.0"}

    @app.get("/health")
    def health() -> dict[str, object]:
        config = minimax_config()
        return {
            "status": "ok",
            "service": "qa-evaluation",
            "judge": "dashscope-minimax",
            "minimax_configured": bool(config.api_key),
            "minimax_model": config.model,
            "minimax_base_url": config.base_url,
            "dashscope_workspace_configured": bool(config.workspace),
        }

    return app


app = create_evaluation_app()


def run_debug_server() -> None:
    parser = argparse.ArgumentParser(description="Run the QA evaluation FastAPI app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("Debug server requires uvicorn: python3 -m pip install uvicorn") from exc

    uvicorn.run("src.main.python.evaluation_main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_debug_server()
