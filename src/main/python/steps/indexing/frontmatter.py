"""Markdown YAML frontmatter 解析器。"""
from __future__ import annotations

import ast
import re
from typing import Any


class FrontmatterParser:
    """YAML frontmatter 解析器。

    解析 ---  ... --- 块中的 YAML 元数据，支持标量、列表和内联结构。
    """

    @staticmethod
    def parse(text: str) -> tuple[dict[str, Any], str]:
        """解析 Markdown 的 frontmatter 和正文。

        Args:
            text: 完整的 Markdown 文本。

        Returns:
            (metadata_dict, body_text) — 无 frontmatter 时返回 ({}, text)。
        """
        if not text.startswith("---\n"):
            return {}, text
        end = text.find("\n---", 4)
        if end == -1:
            return {}, text
        raw = text[4:end].strip()
        body = text[text.find("\n", end + 4) + 1:]
        return FrontmatterParser._parse_yaml(raw), body

    @staticmethod
    def _parse_yaml(raw: str) -> dict[str, Any]:
        """宽松 YAML 解析器：key: value 和缩进列表。"""
        data: dict[str, Any] = {}
        current_key: str | None = None
        list_items: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            # 缩进列表项
            if line.startswith((" ", "\t")) and current_key and line.strip().startswith("- "):
                list_items.append(line.strip()[2:].strip())
                data[current_key] = [FrontmatterParser._parse_scalar(item) for item in list_items]
                continue
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
            if not match:
                current_key = None
                continue
            key, value = match.groups()
            current_key = key
            if value == "":
                list_items = []
                data[key] = []
            else:
                list_items = []
                data[key] = FrontmatterParser._parse_scalar(value.strip())
        return data

    @staticmethod
    def _parse_scalar(value: str) -> Any:
        """类型感知标量解析：bool / None / int / float / list / dict / str。"""
        value = value.strip()
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        if value in {"null", "None", "~"}:
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        if value.startswith("[") or value.startswith("{"):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


# 向后兼容
parse_markdown_with_frontmatter = FrontmatterParser.parse
parse_loose_yaml = FrontmatterParser._parse_yaml
parse_scalar = FrontmatterParser._parse_scalar
