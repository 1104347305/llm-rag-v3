from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    """计算文本的 SHA-256 哈希值（十六进制）。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 哈希值。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

