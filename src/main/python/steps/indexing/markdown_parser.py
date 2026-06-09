"""Markdown 解析器：frontmatter + 正文 + 元数据提取。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.main.python.steps.indexing.frontmatter import FrontmatterParser

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass
class ParsedMarkdown:
    metadata: dict[str, Any]
    body: str
    title: str
    page_type: str
    sources: list[str]
    wikilinks: list[str]


class MarkdownParser:
    """Markdown 文件解析器。

    提取 frontmatter 元数据、标题、来源和 wiki 链接。
    """

    @staticmethod
    def parse(text: str, fallback_title: str) -> ParsedMarkdown:
        """解析 Markdown 文本。

        Args:
            text: 完整 Markdown 文本。
            fallback_title: 无 frontmatter title 时使用的默认标题。

        Returns:
            ParsedMarkdown 对象。
        """
        metadata, body = FrontmatterParser.parse(text)
        title = str(metadata.get("title") or MarkdownParser._first_heading(body) or fallback_title)
        page_type = str(metadata.get("type") or "page")
        sources = MarkdownParser._list_values(metadata.get("sources") or metadata.get("source_files") or [])
        wikilinks = sorted(set(WIKILINK_RE.findall(body) + MarkdownParser._list_values(metadata.get("related") or [])))
        return ParsedMarkdown(metadata, body.strip(), title, page_type, sources, wikilinks)

    @staticmethod
    def _first_heading(body: str) -> str | None:
        """提取正文中第一个 # 标题。"""
        for line in body.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return None

    @staticmethod
    def _list_values(value: Any) -> list[str]:
        """将任意值规范化为字符串列表。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []


# 向后兼容
parse_markdown = MarkdownParser.parse
