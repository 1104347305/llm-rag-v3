from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.main.python.api.routes import _sqlite_fts5_available as sqlite_fts5_available
from src.main.python.config import settings
from src.main.python.services import get_rag_service


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Local RAG index and query CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--project-id", required=True)
    index_parser.add_argument("--path", default="data")
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--build-embeddings", dest="build_embeddings", action="store_true", default=settings.build_embeddings)
    index_parser.add_argument("--no-embeddings", dest="build_embeddings", action="store_false")

    # process-data
    process_parser = subparsers.add_parser("process-data")
    process_parser.add_argument("--project-id", default="pingan-zhenxiang")
    process_parser.add_argument("--path", default="data")
    process_parser.add_argument("--force", action="store_true", default=True)
    process_parser.add_argument("--build-embeddings", dest="build_embeddings", action="store_true", default=settings.build_embeddings)
    process_parser.add_argument("--no-embeddings", dest="build_embeddings", action="store_false")

    # config
    subparsers.add_parser("config", help="Print non-secret runtime configuration.")

    # query
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--project-id", required=True)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--top-pages", type=int, default=settings.default_top_pages)
    query_parser.add_argument("--max-context-size", type=int, default=settings.default_max_context_size)
    query_parser.add_argument("--bm25-top-k", type=int, default=settings.default_bm25_top_k)
    query_parser.add_argument("--vector-top-k", type=int, default=settings.default_vector_top_k)
    query_parser.add_argument("--rerank-top-k", type=int, default=settings.default_rerank_top_k)
    query_parser.add_argument("--no-es", action="store_true", help="Disable ES BM25 recall for this query.")
    query_parser.add_argument("--no-vector", action="store_true", help="Disable vector recall for this query.")
    query_parser.add_argument("--no-lexical", action="store_true", help="Disable local lexical recall for this query.")
    query_parser.add_argument("--debug", action="store_true")

    # answer
    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("--project-id", required=True)
    answer_parser.add_argument("--query", required=True)
    answer_parser.add_argument("--session-id", default=None)
    answer_parser.add_argument("--user-id", default=None)
    answer_parser.add_argument("--history-json", default=None, help="JSON array of {role, content} messages for multi-turn answers.")
    answer_parser.add_argument("--top-pages", type=int, default=settings.default_top_pages)
    answer_parser.add_argument("--max-context-size", type=int, default=settings.default_max_context_size)
    answer_parser.add_argument("--bm25-top-k", type=int, default=settings.default_bm25_top_k)
    answer_parser.add_argument("--vector-top-k", type=int, default=settings.default_vector_top_k)
    answer_parser.add_argument("--rerank-top-k", type=int, default=settings.default_rerank_top_k)
    answer_parser.add_argument("--no-es", action="store_true", help="Disable ES BM25 recall for this query.")
    answer_parser.add_argument("--no-vector", action="store_true", help="Disable vector recall for this query.")
    answer_parser.add_argument("--no-lexical", action="store_true", help="Disable local lexical recall for this query.")

    args = parser.parse_args()
    rag_service = get_rag_service()

    if args.command == "index":
        result = await rag_service.index_project(args.project_id, Path(args.path), force=args.force, build_embeddings=args.build_embeddings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "process-data":
        result = await rag_service.index_project(args.project_id, Path(args.path), force=args.force, build_embeddings=args.build_embeddings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "config":
        print(json.dumps({
            "env": "dev",
            "llm_model": settings.llm_model,
            "has_dashscope_api_key": bool(settings.dashscope_api_key),
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "storage_dir": str(settings.storage_dir),
            "es_configured": settings.es_enabled,
            "es_indexing_enabled": settings.es_indexing_enabled,
            "es_retrieval_enabled": settings.es_retrieval_enabled,
            "pg_host": settings.pg_host,
            "pgvector_enabled": settings.pgvector_enabled,
            "enable_vector_retrieval": settings.enable_vector_retrieval,
            "enable_local_lexical": settings.enable_local_lexical_retrieval,
            "build_embeddings": settings.build_embeddings,
            "default_top_pages": settings.default_top_pages,
            "log_level": settings.log_level,
            "enable_agno_agent": settings.enable_agno_agent,
            "sqlite_fts5_available": sqlite_fts5_available(),
        }, ensure_ascii=False, indent=2))
    elif args.command == "query":
        result = await rag_service.retrieve_context(
            args.project_id, args.query,
            top_pages=args.top_pages, max_context_size=args.max_context_size,
            bm25_top_k=args.bm25_top_k, vector_top_k=args.vector_top_k,
            rerank_top_k=args.rerank_top_k,
            include_es=False if args.no_es else None,
            include_vector=False if args.no_vector else None,
            include_lexical=False if args.no_lexical else None,
            debug=args.debug,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "answer":
        history = json.loads(args.history_json) if args.history_json else None
        result = await rag_service.answer_query(
            args.project_id, args.query,
            session_id=args.session_id, user_id=args.user_id, history=history,
            top_pages=args.top_pages, max_context_size=args.max_context_size,
            bm25_top_k=args.bm25_top_k, vector_top_k=args.vector_top_k,
            rerank_top_k=args.rerank_top_k,
            include_es=False if args.no_es else None,
            include_vector=False if args.no_vector else None,
            include_lexical=False if args.no_lexical else None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
