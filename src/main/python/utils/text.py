from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """英文分词：小写 + 按非字母数字字符分割 + 过滤空串。"""

    return [token.lower() for token in TOKEN_RE.findall(text)]


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """估算 token 数，区分中英文字符。

    比例从 config 读取，默认中文 ~1.5 char/token，英文 ~4 char/token。
    """
    if not text:
        return 1
    from src.main.python.config import settings
    cjk_chars = len(_CJK_RE.findall(text))
    ascii_chars = len(text) - cjk_chars
    estimated = cjk_chars / settings.token_estimate_cjk_ratio + ascii_chars / settings.token_estimate_ascii_ratio
    return max(1, math.ceil(estimated))


def term_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))

