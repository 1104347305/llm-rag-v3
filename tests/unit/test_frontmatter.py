import unittest

from src.main.python.steps.indexing.frontmatter import parse_markdown_with_frontmatter


class TestFrontmatter(unittest.TestCase):
    def test_parse_current_data_frontmatter_lists_and_scalars(self):
        text = """---
title: "家庭医生服务"
type: entity
tags: ["臻享家医", "家庭医生"]
needs_review: true
---
# 家庭医生服务
"""
        metadata, body = parse_markdown_with_frontmatter(text)
        assert metadata["title"] == "家庭医生服务"
        assert metadata["type"] == "entity"
        assert metadata["tags"] == ["臻享家医", "家庭医生"]
        assert metadata["needs_review"] is True
        assert body.startswith("# 家庭医生服务")
