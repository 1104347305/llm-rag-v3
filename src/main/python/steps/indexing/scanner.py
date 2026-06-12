from __future__ import annotations

import asyncio
from pathlib import Path


def _knowledge_root(project_path: Path) -> Path:
    if (project_path / "wiki").is_dir():
        return project_path / "wiki"
    if (project_path / "data").is_dir():
        return project_path / "data"
    return project_path


def _scan_markdown_files_sync(project_path: Path) -> list[Path]:
    root = _knowledge_root(project_path)
    return sorted(
        path for path in root.rglob("*.md")
        if path.is_file() and ".rag" not in path.parts
    )


async def scan_markdown_files(project_path: Path) -> list[Path]:
    """异步扫描知识库中的 Markdown 文件。"""
    return await asyncio.to_thread(_scan_markdown_files_sync, project_path)
