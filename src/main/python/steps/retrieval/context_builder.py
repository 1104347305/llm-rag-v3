"""上下文构建器——将检索结果组装为 LLM 可用的格式化上下文。"""
from __future__ import annotations

from src.main.python.config import settings
from src.main.python.models import Chunk, Page


class ContextBuilder:
    """从检索结果构建 LLM 上下文。

    build() 将页面和 chunk 按字符预算打包为 markdown 格式，
    包含页面列表、索引和分页内容。
    """

    def __init__(self) -> None:
        """初始化上下文构建器。"""

        pass

    def build(self, selected: list[dict[str, object]],
              chunks_by_id: dict[str, Chunk],
              pages_by_id: dict[str, Page],
              max_context_size: int) -> dict[str, object]:
        """构建 LLM 上下文。

        Args:
            selected: 评分选中的页面列表 [{"page_id", "chunk_ids", "score", "source"}, ...]。
            chunks_by_id: chunk_id → Chunk 查找表。
            pages_by_id: page_id → Page 查找表。
            max_context_size: 最大上下文字符预算。

        Returns:
            dict，包含:
            - pages: 页面元数据列表（不含完整 text）
            - page_list: 页面编号列表字符串 "[1] Title (path)"
            - pages_context: 合并的 markdown 格式内容
            - index: 截断后的 page_list
            - source: 硬编码的检索来源标识
        """
        max_ctx = max_context_size if max_context_size and max_context_size > 0 else settings.default_max_context_size

        # 预算分配：index_budget 给索引列表，page_budget 给页面内容
        index_budget = int(max_ctx * settings.index_budget_fraction)
        page_budget = int(max_ctx * settings.page_budget_fraction)

        # 单页上限 = min(总页面预算, max(底限, 页面预算 × 比例))
        per_page_cap = min(page_budget, max(
            settings.per_page_budget_floor,
            int(page_budget * settings.per_page_budget_fraction),
        ))

        used = 0
        page_blocks: list[dict[str, object]] = []
        page_list_lines: list[str] = []

        for number, item in enumerate(selected, start=1):
            page = pages_by_id[str(item["page_id"])]
            chunk_ids = [cid for cid in item.get("chunk_ids", []) if cid in chunks_by_id]
            page_chunks = [chunks_by_id[cid] for cid in chunk_ids]

            remaining = max(0, page_budget - used)
            allowed = min(per_page_cap, remaining)
            if allowed <= 0:
                continue

            content = self._pack_page(page, page_chunks, allowed)
            used += len(content)

            page_list_lines.append(f"[{number}] {page.title} ({page.path})")
            page_blocks.append({
                "number": number,
                "page_id": page.page_id,
                "title": page.title,
                "path": page.path,
                "score": item.get("score", 0.0),
                "source": item.get("source", "hybrid"),
                "chunks": [
                    {"chunk_id": c.chunk_id, "heading_path": c.heading_path,
                     "content": c.content, "score": item.get("score", 0.0)}
                    for c in page_chunks
                ],
                "text": content,
            })

        # 索引截断
        raw_index = "\n".join(page_list_lines)
        index = raw_index[:index_budget]
        if len(raw_index) > index_budget:
            index += "\n\n[...index trimmed...]"

        # 合并页面内容
        pages_context = "\n\n---\n\n".join(
            f"### [{b['number']}] {b['title']}\nPath: {b['path']}\n\n{b['text']}"
            for b in page_blocks
        )

        return {
            "pages": [{k: v for k, v in b.items() if k != "text"} for b in page_blocks],
            "page_list": "\n".join(page_list_lines),
            "pages_context": pages_context,
            "index": index,
            "source": "page_context_es_bm25_text_embedding_v4_rrf_graph_qwen3_rerank",
        }

    # ── 页面打包 ────────────────────────────────────────────

    def _pack_page(self, page: Page, anchor_chunks: list[Chunk], cap: int) -> str:
        """将页面内容打包到 cap 字符内。

        策略：
        1. 内容 ≤ cap → 直接返回
        2. 内容 ≤ section_first_threshold 或无 anchor → 截断
        3. 否则 → 以 anchor chunks 为中心提取 section 窗口
        """
        if len(page.content) <= cap:
            return page.content
        if len(page.content) <= settings.section_first_threshold or not anchor_chunks:
            return ContextBuilder._truncate(page.content, cap)

        # 以每个 anchor chunk 为中心提取上下文窗口
        sections: list[str] = []
        for chunk in anchor_chunks:
            section = ContextBuilder._section_window(
                page.content, chunk.content, settings.section_neighbor_chars)
            if section and section not in sections:
                sections.append(section)

        # 页面开头前缀 + section 窗口
        prefix = page.content[:min(settings.page_prefix_chars, cap)]
        parts = sections + [f"### Page Opening\n{prefix}"]

        packed_parts: list[str] = []
        packed_len = 0
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            addition = part if i == 0 or not packed_parts else f"\n\n---\n\n{part}"
            remaining = cap - packed_len
            if remaining <= 0:
                break
            if len(addition) > remaining:
                packed_parts.append(addition[:remaining])
                break
            packed_parts.append(addition)
            packed_len += len(addition)
        return ContextBuilder._truncate("".join(packed_parts), cap)

    # ── 静态辅助方法 ────────────────────────────────────────

    @staticmethod
    def _section_window(page_content: str, chunk_content: str, neighbor_chars: int) -> str:
        """在页面中定位 chunk 并提取包含前后 neighbor_chars 字符的上下文窗口。

        如果精确匹配失败，用前 section_search_fragment_chars 个字符模糊匹配。
        仍失败则返回 chunk_content 自身。
        """
        needle = chunk_content.strip()
        if not needle:
            return ""
        idx = page_content.find(needle)
        if idx < 0:
            # 模糊匹配：只用前 N 个字符搜索
            needle = needle[:min(len(needle), settings.section_search_fragment_chars)].strip()
            idx = page_content.find(needle) if needle else -1
        if idx < 0:
            return chunk_content
        start = max(0, idx - neighbor_chars)
        end = min(len(page_content), idx + len(needle) + neighbor_chars)
        return page_content[start:end].strip()

    @staticmethod
    def _truncate(text: str, cap: int) -> str:
        """截断文本到 cap 字符，超出部分附加 [...truncated...] 标记。"""
        if len(text) <= cap:
            return text
        suffix = "\n\n[...truncated...]"
        if cap <= len(suffix):
            return text[:cap]
        return text[:cap - len(suffix)] + suffix
