from __future__ import annotations

__all__ = ["RAGService", "get_rag_service"]


def __getattr__(name: str):
    if name in __all__:
        from src.main.python.services.rag_service import (
            RAGService,
            get_rag_service,
        )

        return {
            "RAGService": RAGService,
            "get_rag_service": get_rag_service,
        }[name]
    raise AttributeError(name)
