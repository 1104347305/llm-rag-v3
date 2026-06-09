"""基于知识图谱的页面扩展。

从已检索页面沿 GraphEdge 边扩展到语义相关页面。
"""
from __future__ import annotations

from collections import defaultdict

from src.main.python.config import settings
from src.main.python.models import GraphEdge


class GraphExpander:
    """基于知识图谱的页面扩展器。

    从 page_results 中取前 limit_pages 个源页面，
    沿 GraphEdge 边（按 weight 降序）扩展到目标页面。
    过滤条件：weight >= 2.0，每页最多 per_page 个新页面，不重复。

    使用示例:
        expander = GraphExpander()
        expansions = expander.expand(page_results, edges)
        # expansions: [{"page_id", "source_page_id", "graph_score", "edge_type"}, ...]
    """

    def expand(self,
               page_results: list[dict[str, object]],
               edges: list[GraphEdge],
               limit_pages: int | None = None,
               per_page: int | None = None) -> list[dict[str, object]]:
        """从已检索页面沿边扩展到相关页面。

        Args:
            page_results: 已检索的页面列表（按分数排序），每项含 "page_id" 键。
            edges: 图边列表，每条边包含 source_page_id, target_page_id, weight, edge_type。
            limit_pages: 最多从多少个源页面出发扩展。默认取自 settings.graph_expand_limit_pages。
            per_page: 每页最多引入多少个新页面。默认取自 settings.graph_expand_per_page。

        Returns:
            扩展页面列表，每项包含:
            - page_id: 目标页面 ID
            - source_page_id: 来源页面 ID
            - graph_score: 边的权重
            - edge_type: 边类型（如 wikilink, shared_source 等）
        """
        limit_pages = limit_pages if limit_pages is not None else settings.graph_expand_limit_pages
        per_page = per_page if per_page is not None else settings.graph_expand_per_page

        # 已访问页面集合（避免重复扩展）
        seen = {str(item["page_id"]) for item in page_results}

        # 构建邻接表：source_page_id → 所有出边
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source_page_id].append(edge)

        expansions: list[dict[str, object]] = []
        for item in page_results[:limit_pages]:
            source = str(item["page_id"])
            # 按边权重降序排列，优先扩展强关联页面
            candidates = sorted(adjacency.get(source, []), key=lambda edge: edge.weight, reverse=True)
            added = 0
            for edge in candidates:
                # 跳过已访问页面和弱关联边（weight < 2.0）
                if edge.target_page_id in seen or edge.weight < 2.0:
                    continue
                expansions.append(
                    {
                        "page_id": edge.target_page_id,
                        "source_page_id": source,
                        "graph_score": edge.weight,
                        "edge_type": edge.edge_type,
                    }
                )
                seen.add(edge.target_page_id)
                added += 1
                if added >= per_page:
                    break
        return expansions
