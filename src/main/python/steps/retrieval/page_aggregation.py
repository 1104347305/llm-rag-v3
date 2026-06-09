from __future__ import annotations

from collections import defaultdict

from src.main.python.config import settings
from src.main.python.models import Chunk, Page


def aggregate_pages(fused_chunks: list[dict[str, object]], chunks_by_id: dict[str, Chunk], pages_by_id: dict[str, Page]) -> list[dict[str, object]]:
    """将 RRF 融合的 chunks 按 page_id 分组打分，返回页面排序列表。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in fused_chunks:
        chunk = chunks_by_id.get(str(item["id"]))
        if chunk:
            grouped[chunk.page_id].append(item)

    _type_boost = {
        "entity": settings.page_type_entity_boost,
        "concept": settings.page_type_concept_boost,
        "source": settings.page_type_source_boost,
    }

    pages: list[dict[str, object]] = []
    for page_id, items in grouped.items():
        page = pages_by_id[page_id]
        scores = sorted([float(item["score"]) for item in items], reverse=True)
        matched_chunks = [chunks_by_id[str(item["id"])] for item in items if str(item["id"]) in chunks_by_id]
        title_boost = settings.page_title_boost if any(page.title in chunk.content for chunk in matched_chunks) else 0.0
        tail = sum(scores[1:5])
        page_score = scores[0] + settings.page_tail_weight * tail + title_boost
        _boost = _type_boost.get(page.type)
        if _boost is not None:
            page_score *= _boost
        anchor_count = settings.anchor_chunks_per_page
        anchor_chunks = [str(item["id"]) for item in sorted(items, key=lambda item: float(item["score"]), reverse=True)[:anchor_count]]
        sources = sorted({source for item in items for source in item.get("sources", [])})
        pages.append(
            {
                "page_id": page_id,
                "title": page.title,
                "path": page.path,
                "score": page_score,
                "anchor_chunks": anchor_chunks,
                "matched_sources": sources,
            }
        )
    return sorted(pages, key=lambda item: float(item["score"]), reverse=True)
