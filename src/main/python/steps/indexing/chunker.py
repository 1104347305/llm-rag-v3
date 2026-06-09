"""Markdown 文档分块器——将 Page 拆分为语义连贯的 Chunk。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.main.python.config import settings
from src.main.python.models import Chunk, Page
from src.main.python.utils.text import estimate_tokens


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = settings.chunk_target_chars
    max_chars: int = settings.chunk_max_chars
    min_chars: int = settings.chunk_min_chars
    overlap_chars: int = settings.chunk_overlap_chars
    merge_strategy: str = settings.chunk_merge_strategy
    standalone_threshold: int = settings.chunk_standalone_threshold


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class DocumentChunker:
    """Markdown 文档分块器。

    流程：按标题切分 → 按空行切块 → 合并 → 添加重叠 → 构建 Chunk 对象。
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """初始化分块器。"""

        self.config = config or ChunkingConfig()

    def chunk(self, page: Page) -> list[Chunk]:
        """将 Page 拆分为 Chunk 列表。

        Args:
            page: 已解析的 Page 对象。

        Returns:
            按 chunk_index 排序的 Chunk 列表，含 prev/next 链。
        """
        parts = self._split_sections(page)

        # 按空行切块 → 合并
        merged = self._merge_blocks(parts)

        # 添加前后重叠
        merged = self._add_overlap(merged)

        # 构建 Chunk 对象
        chunks: list[Chunk] = []
        for index, (heading_path, content) in enumerate(merged):
            chunk_id = f"{page.page_id}#{index:04d}"
            chunks.append(Chunk(
                project_id=page.project_id,
                page_id=page.page_id,
                chunk_id=chunk_id,
                path=page.path,
                title=page.title,
                heading_path=heading_path,
                type=page.type,
                sources=page.sources,
                content=content,
                chunk_index=index,
                token_estimate=estimate_tokens(content),
            ))

        # 构建双向链表
        for index, chunk in enumerate(chunks):
            chunk.prev_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
            chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
        return chunks

    # ── 切分 ────────────────────────────────────────────────

    def _split_sections(self, page: Page) -> list[tuple[str, str]]:
        """按 markdown 标题切分为 (heading_path, text) 对。"""
        lines = page.content.splitlines()
        sections: list[tuple[str, list[str]]] = []
        heading_stack: list[tuple[int, str]] = [(1, page.title)]
        current: list[str] = []

        def heading_path() -> str:
            names = [name for _, name in heading_stack]
            return " > ".join(dict.fromkeys(names))

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                if current:
                    sections.append((heading_path(), current))
                    current = []
                level = len(match.group(1))
                name = match.group(2).strip()
                heading_stack[:] = [(lvl, text) for lvl, text in heading_stack if lvl < level]
                heading_stack.append((level, name))
            current.append(line)
        if current:
            sections.append((heading_path(), current))

        return [(heading, "\n".join(section).strip()) for heading, section in sections]

    def _merge_blocks(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """将 (heading, block) 对合并，过长的 block 先拆分。"""
        # 先拆分过长的 block
        flat: list[tuple[str, str]] = []
        for heading_path, section_text in parts:
            for block in self._split_blocks(section_text):
                if block.strip():
                    flat.append((heading_path, block.strip()))

        # 合并相邻小块
        merged: list[tuple[str, str]] = []
        for heading_path, block in flat:
            if self._should_start_new_chunk(merged, heading_path, block):
                merged.append((heading_path, block))
            else:
                last_heading, last_text = merged[-1]
                heading = last_heading if last_heading == heading_path else heading_path
                merged[-1] = (heading, f"{last_text}\n\n{block}")
        return merged

    @staticmethod
    def _split_blocks(text: str, max_chars: int | None = None) -> list[str]:
        """按空行切分为 block，过长的进一步拆分。"""
        max_chars = max_chars or settings.chunk_max_chars
        blocks = re.split(r"\n\s*\n", text)
        output: list[str] = []
        for block in blocks:
            block = block.strip()
            if len(block) <= max_chars:
                output.append(block)
                continue
            output.extend(DocumentChunker._split_long_block(block, max_chars))
        return output

    @staticmethod
    def _split_long_block(block: str, max_chars: int) -> list[str]:
        """拆分超长 block：表格按行、普通文本按句号、否则按字符。"""
        if "\n|" in block or block.startswith("|"):
            return DocumentChunker._split_by_lines(block, max_chars)
        sentences = re.split(r"(?<=[。！？!?；;])", block)
        if len(sentences) > 1:
            return DocumentChunker._pack_units(sentences, max_chars)
        return [block[i:i + max_chars] for i in range(0, len(block), max_chars)]

    @staticmethod
    def _split_by_lines(text: str, max_chars: int) -> list[str]:
        """按行拆分文本并打包。"""

        return DocumentChunker._pack_units([line + "\n" for line in text.splitlines()], max_chars)

    @staticmethod
    def _pack_units(units: list[str], max_chars: int) -> list[str]:
        """将文本单元打包到不超过 max_chars 的组中。"""
        output: list[str] = []
        current = ""
        for unit in units:
            if current and len(current) + len(unit) > max_chars:
                output.append(current.strip())
                current = unit
            else:
                current += unit
        if current.strip():
            output.append(current.strip())
        return output

    # ── 合并策略 ────────────────────────────────────────────

    def _should_start_new_chunk(self, merged: list[tuple[str, str]], heading_path: str, block: str) -> bool:
        """判断是否开始新 chunk。

        auto: 超 target_chars 才分块。
        independent: 大块独立（≥standalone_threshold），小块合并。
        """
        if not merged:
            return True

        last_len = len(merged[-1][1])
        block_len = len(block)

        if last_len + block_len > self.config.target_chars:
            return True

        if self.config.merge_strategy == "independent":
            if block_len >= self.config.standalone_threshold:
                return True
            if last_len >= self.config.standalone_threshold:
                return True

        return False

    # ── 重叠 ────────────────────────────────────────────────

    def _add_overlap(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """相邻块之间添加 overlap_chars 的重叠文本。"""
        overlap_chars = self.config.overlap_chars
        if overlap_chars <= 0 or len(parts) <= 1:
            return parts
        output = [parts[0]]
        for index in range(1, len(parts)):
            heading_path, content = parts[index]
            previous = parts[index - 1][1]
            overlap = previous[-overlap_chars:].strip()
            if overlap:
                content = f"{overlap}\n\n{content}"
            output.append((heading_path, content))
        return output


# 向后兼容
chunk_page = DocumentChunker().chunk
