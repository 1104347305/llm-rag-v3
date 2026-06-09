"""知识图谱边构建器——基于 wikilink、shared_source、same_type 构建图边。"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from src.main.python.config import settings
from src.main.python.models import GraphEdge, Page


class GraphBuilder:
    """图边构建器。

    三条边类型：
    - wikilink: [[page_name]] 引用，weight=3.0
    - shared_source: 同 source 的页面两两连接，weight=4.0
    - same_type: 同 type 的页面两两连接，weight=1.0

    大组（超过阈值）使用确定性哈希采样防止 O(n²) 爆炸。
    """

    @staticmethod
    def build(project_id: str, pages: list[Page]) -> list[GraphEdge]:
        """从页面列表构建图边。

        Args:
            project_id: 项目标识。
            pages: 页面列表。

        Returns:
            GraphEdge 列表，已去重并按 (source, target, type) 排序。
        """
        by_title = {page.title: page.page_id for page in pages}
        by_page_id = {page.page_id: page for page in pages}
        edges: dict[tuple[str, str, str], float] = {}

        def add(source: str, target: str, edge_type: str, weight: float) -> None:
            if source == target:
                return
            key = (source, target, edge_type)
            edges[key] = max(edges.get(key, 0.0), weight)

        # wikilink 边
        for page in pages:
            for link in page.wikilinks:
                target = by_title.get(link) or by_page_id.get(link)
                if isinstance(target, str):
                    add(page.page_id, target, "wikilink", 3.0)

        # shared_source 边
        sources_index: dict[str, list[str]] = defaultdict(list)
        for page in pages:
            for source in page.sources:
                sources_index[source].append(page.page_id)
        for page_ids in sources_index.values():
            if len(page_ids) <= settings.graph_max_shared_source_links:
                for source in page_ids:
                    for target in page_ids:
                        add(source, target, "shared_source", 4.0)
            else:
                for page_id in page_ids:
                    neighbours = GraphBuilder._sample(page_id, page_ids, settings.graph_max_shared_source_links)
                    for target in neighbours:
                        add(page_id, target, "shared_source", 4.0)

        # same_type 边
        type_index: dict[str, list[str]] = defaultdict(list)
        for page in pages:
            type_index[page.type].append(page.page_id)
        for page_ids in type_index.values():
            if len(page_ids) <= settings.graph_max_same_type_links:
                for source in page_ids:
                    for target in page_ids:
                        add(source, target, "same_type", 1.0)
            else:
                for page_id in page_ids:
                    neighbours = GraphBuilder._sample(page_id, page_ids, settings.graph_max_same_type_links)
                    for target in neighbours:
                        add(page_id, target, "same_type", 1.0)

        return [
            GraphEdge(project_id=project_id, source_page_id=source, target_page_id=target,
                       edge_type=edge_type, weight=weight)
            for (source, target, edge_type), weight in sorted(edges.items())
        ]

    @staticmethod
    def _sample(page_id: str, all_ids: list[str], k: int) -> list[str]:
        """确定性采样 k 个邻居（基于 page_id 哈希，不含自身）。"""
        if len(all_ids) <= k + 1:
            return [pid for pid in all_ids if pid != page_id]
        seed = int(hashlib.md5(page_id.encode()).hexdigest(), 16)
        available = list(range(len(all_ids)))
        sampled: list[str] = []
        state = seed
        while len(sampled) < k and available:
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            idx = state % len(available)
            candidate = all_ids[available[idx]]
            if candidate != page_id:
                sampled.append(candidate)
            available.pop(idx)
        return sampled


# 向后兼容
build_graph_edges = GraphBuilder.build
