# ============================================================
# 日志系统 — 基于 Loguru 的统一日志
# ============================================================
# 用法:
#   logger = get_logger(__name__)
#   log_event(logger, 20, "event.name", "描述", key=val, ...)
#   级别数字: 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR
#
# 输出格式:
#   时间 | 级别 | 模块名 | 事件名 | 消息 | key=value ...
# 敏感字段（api_key, password 等）自动过滤
# ============================================================

from __future__ import annotations

import json
import sys
from typing import Any

from loguru import logger as _loguru_root

from src.main.python.config import settings

_CONFIGURED = False

# Loguru 格式模板：{name} 为内置模块名
_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name: <32} |"
    " {message}"
)


def configure_logging() -> None:
    """配置 Loguru sink（仅首次调用生效）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _loguru_root.remove()
    _loguru_root.add(sys.stderr, level=settings.log_level, format=_FMT, colorize=True)

    if settings.log_file:
        _loguru_root.add(
            settings.log_file, level=settings.log_level, format=_FMT,
            rotation="10 MB", retention="7 days", encoding="utf-8",
        )

    _CONFIGURED = True


def get_logger(name: str):
    """返回绑定模块名的 Loguru logger。"""
    configure_logging()
    return _loguru_root.bind(name=name, event="")


def log_event(logger, level: int, event: str, message: str, **fields: Any) -> None:
    """记录结构化日志事件。

    extra 字段序列化后追加到 message 尾部，自动过滤敏感信息。
    """
    level_map = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
    level_name = level_map.get(level, "INFO")

    clean = sanitize_log_value(fields)
    parts = [message]
    for k, v in clean.items():
        parts.append(f"{k}={_format_value(v)}")

    logger.bind(event=event).log(level_name, " | ".join(parts))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value - int(value)) < 0.001 else f"{value:.1f}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ── 敏感信息过滤 ──────────────────────────────────────────

def sanitize_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_log_value(v) for k, v in value.items() if not _is_secret_key(str(k))}
    if isinstance(value, (list, tuple)):
        return [sanitize_log_value(item) for item in value]
    if isinstance(value, str) and _looks_like_secret(value):
        return "***"
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(t in lowered for t in ("api_key", "apikey", "token", "secret", "password", "authorization"))


def _looks_like_secret(value: str) -> bool:
    return value.startswith("sk-") and len(value) > 8
