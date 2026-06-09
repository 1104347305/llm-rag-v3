# ============================================================
# 计时工具 — 用于管道各步骤的耗时统计
# ============================================================
# timer(metrics, key): 纯计时，结果写入 metrics[key]
# trace(metrics, key, logger, step, input, output): 计时 + 日志（打印输入/输出摘要）
# ============================================================

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator


@contextmanager
def timer(metrics: dict[str, float], key: str) -> Iterator[None]:
    """纯计时上下文管理器，结果写入 metrics[key]（毫秒）。"""
    start = perf_counter()
    try:
        yield
    finally:
        metrics[key] = round((perf_counter() - start) * 1000, 3)


@contextmanager
def trace(
    metrics: dict[str, float],
    key: str,
    logger,
    step: str,
    *,
    input: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """计时 + 打印步骤日志。

    用法:
        with trace(metrics, "bm25", logger, "BM25检索",
                   input=dict(query=q, top_k=k)) as out:
            results = ...
            out["count"] = len(results)
            out["top_score"] = results[0][1] if results else 0
    """
    result: dict[str, Any] = {}
    start = perf_counter()
    try:
        yield result
    finally:
        latency = round((perf_counter() - start) * 1000, 3)
        metrics[key] = latency

        parts = [f"{latency}ms | {step}"]
        if input:
            parts.append(" | in: " + _format_summary(input))
        merged = {**result, **(output or {})}
        if merged:
            parts.append(" | out: " + _format_summary(merged))

        logger.bind(event="trace").info("".join(parts))


def _format_summary(data: dict[str, Any], max_keys: int = 6) -> str:
    """格式化摘要字典，最多显示 max_keys 个字段。"""
    items = [(k, v) for k, v in data.items() if v is not None]
    shown = items[:max_keys]
    extra = f" ...+{len(items) - max_keys}" if len(items) > max_keys else ""
    return " ".join(f"{k}={_format_val(v)}" for k, v in shown) + extra


def _format_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, str)):
        s = str(v)
        return s[:60] + "..." if len(s) > 60 else s
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        if len(v) <= 5:
            items = []
            for x in v:
                if isinstance(x, float) and abs(x) < 10:
                    items.append(f"{x:.3f}")
                elif isinstance(x, str):
                    items.append(x[:20])
                else:
                    items.append(str(x))
            return "[" + ",".join(items) + "]"
        return f"[{len(v)}]"
    if isinstance(v, dict):
        return f"{{{len(v)}}}"
    return str(v)[:40]
