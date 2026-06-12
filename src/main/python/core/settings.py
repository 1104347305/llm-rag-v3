# ============================================================
# 环境感知配置系统
# ============================================================
# 配置优先级（从高到低）:
#   环境变量 > YAML 配置文件 > 代码默认值
#
# YAML 配置文件按 ENV 环境变量自动选择:
#   ENV=dev  → src/main/python/config/dev_app_args.yaml
#   ENV=stg  → src/main/python/config/stg_app_args.yaml
#   ENV=prd  → src/main/python/config/prd_app_args.yaml
#
# .env 文件通过 _load_dotenv() 加载，仅设置未被显式定义的环境变量
# ============================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # PyYAML 未安装时回退到默认值


# ── .env 文件加载 ──────────────────────────────────────────
def _load_dotenv(path: Path = Path(".env")) -> None:
    """加载 .env 文件到 os.environ。

    注意：仅 setdefault，不覆盖已存在的环境变量。
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# ── YAML 配置路径 ──────────────────────────────────────────
# 默认从 settings.py 所在目录查找，不受 CWD 影响
_ENV = os.getenv("ENV", "dev").lower()
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_YAML_PATH = Path(os.getenv("RAG_CONFIG_DIR", str(_DEFAULT_CONFIG_DIR))) / f"{_ENV}_app_args.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件，失败时返回 {}。"""

    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_YAML_CONFIG = _load_yaml_config(_YAML_PATH)


# ── 配置读取辅助函数 ───────────────────────────────────────
# 优先级: 环境变量 > YAML > 默认值
def _config_value(section: str, key: str, env_name: str, default: Any) -> Any:
    """按优先级读取配置值：YAML > 环境变量 > 默认值。"""
    section_values = _YAML_CONFIG.get(section, {})
    if isinstance(section_values, dict) and key in section_values:
        return section_values[key]
    raw = os.getenv(env_name)
    if raw is not None:
        return raw
    return default


def _config_str(section: str, key: str, env_name: str, default: str = "") -> str:
    """按优先级读取字符串配置：环境变量 > YAML > 默认值。"""

    return str(_config_value(section, key, env_name, default))


def _config_int(section: str, key: str, env_name: str, default: int) -> int:
    """按优先级读取整型配置：环境变量 > YAML > 默认值。"""

    return int(_config_value(section, key, env_name, default))


def _config_bool(section: str, key: str, env_name: str, default: bool) -> bool:
    """按优先级读取布尔配置：环境变量 > YAML > 默认值。"""

    raw = os.getenv(env_name)
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off", ""}
    value = _config_value(section, key, env_name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


# ── Settings 主类 ──────────────────────────────────────────
@dataclass(frozen=True)
class Settings:
    """全局配置（frozen dataclass，线程安全）。

    配置项按 YAML section 分组。每个字段均注明：
      - YAML 路径（section.key）
      - 对应环境变量
      - 默认值
      - 用途说明
    """

    # ═══════════════════════════════════════════════════════
    # server — HTTP 服务
    # ═══════════════════════════════════════════════════════
    host: str = _config_str("server", "host", "RAG_HOST", "0.0.0.0")
    """FastAPI 监听地址。0.0.0.0 表示接受所有网络接口的连接。"""

    port: int = _config_int("server", "port", "RAG_PORT", 8010)
    """FastAPI 监听端口，默认 8010。"""

    # ═══════════════════════════════════════════════════════
    # paths — 文件和存储路径
    # ═══════════════════════════════════════════════════════
    default_data_path: Path = Path(_config_str("paths", "default_data_path", "WIKI_DATA_PATH", "data"))
    """Markdown 知识库根目录，索引和检索都从这里读取 .md 文件。"""

    storage_dir: Path = Path(_config_str("paths", "storage_dir", "RAG_STORAGE_DIR", ".rag"))
    """本地索引数据存放目录。包含 JSON 快照、SQLite 数据库、manifest 文件。"""

    # ═══════════════════════════════════════════════════════
    # models — AI 模型选择
    # ═══════════════════════════════════════════════════════
    embedding_model: str = _config_str("models", "embedding_model", "EMBEDDING_MODEL", "text-embedding-v4")
    """DashScope 嵌入模型名称，用于将文本转为向量。"""

    embedding_dim: int = _config_int("models", "embedding_dim", "EMBEDDING_DIM", 0)
    """向量维度。0 表示自动检测（DashScope API 返回的维度）。手动设置可覆盖。"""

    rerank_model: str = _config_str("models", "rerank_model", "RERANK_MODEL", "qwen3-rerank")
    """DashScope 重排序模型，用于对候选 chunk 精排打分。"""

    llm_model: str = _config_str("models", "llm_model", "LLM_MODEL", "qwen3.6-35b-a3b")
    """大语言模型名称，用于最终答案生成。"""

    llm_reasoning_effort: str = _config_str("models", "llm_reasoning_effort", "LLM_REASONING_EFFORT", "")
    """LLM 推理深度。空字符串表示使用默认值。可设为 low / medium / high。"""

    # ═══════════════════════════════════════════════════════
    # dashscope — 阿里云 DashScope API 端点
    # ═══════════════════════════════════════════════════════
    dashscope_api_key: str = _config_str("dashscope", "api_key", "DASHSCOPE_API_KEY", "")
    """DashScope API 密钥。空字符串 = 嵌入/重排序/LLM 均不可用，回退到本地降级方案。"""

    dashscope_embedding_base_url: str = _config_str(
        "dashscope", "embedding_base_url", "DASHSCOPE_EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    """DashScope 嵌入 API 的基础 URL。"""

    dashscope_rerank_base_url: str = _config_str(
        "dashscope", "rerank_base_url", "DASHSCOPE_RERANK_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-api/v1"
    )
    """DashScope 重排序 API 的基础 URL。"""

    dashscope_chat_base_url: str = _config_str(
        "dashscope", "chat_base_url", "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    """DashScope LLM 对话 API 的基础 URL（OpenAI 兼容接口）。"""

    internal_embedding_url: str = _config_str(
        "internal_models", "embedding_url", "INTERNAL_EMBEDDING_URL", ""
    )
    """内网 embedding POST 地址。配置后优先于 DashScope embedding。"""

    internal_rerank_url: str = _config_str(
        "internal_models", "rerank_url", "INTERNAL_RERANK_URL", ""
    )
    """内网 rerank POST 地址。配置后优先于 DashScope rerank。"""

    internal_model_timeout: int = _config_int(
        "internal_models", "request_timeout", "INTERNAL_MODEL_TIMEOUT", 30
    )
    """内网 embedding/rerank HTTP 请求超时秒数。"""

    internal_model_max_connections: int = _config_int(
        "internal_models", "max_connections", "INTERNAL_MODEL_MAX_CONNECTIONS", 32
    )
    """内网模型共享 HTTP 客户端最大连接数。"""

    # ═══════════════════════════════════════════════════════
    # postgres — pgvector 向量数据库
    # ═══════════════════════════════════════════════════════
    pg_host: str = _config_str("postgres", "host", "PG_HOST", "localhost")
    """PostgreSQL 主机地址。空字符串 = pgvector 不可用。"""

    pg_port: int = _config_int("postgres", "port", "PG_PORT", 5432)
    """PostgreSQL 端口，默认 5432。"""

    pg_database: str = _config_str("postgres", "database", "PG_DATABASE", "rag")
    """PostgreSQL 数据库名。"""

    pg_user: str = _config_str("postgres", "user", "PG_USER", "postgres")
    """PostgreSQL 用户名。"""

    pg_password: str = _config_str("postgres", "password", "PG_PASSWORD", "")
    """PostgreSQL 密码。"""

    pg_vector_ef_search: int = _config_int("postgres", "vector_ef_search", "PG_VECTOR_EF_SEARCH", 200)
    """HNSW 索引查询时的搜索深度（ef_search）。值越大精度越高、速度越慢。"""

    # ═══════════════════════════════════════════════════════
    # elasticsearch — BM25 词汇检索
    # ═══════════════════════════════════════════════════════
    es_url: str = _config_str("elasticsearch", "url", "ES_URL", "")
    """Elasticsearch 服务地址。空字符串 = ES 不可用。"""

    es_user: str = _config_str("elasticsearch", "user", "ES_USER", "")
    """ES 用户名（Basic Auth）。空字符串 = 无认证。"""

    es_password: str = _config_str("elasticsearch", "password", "ES_PASSWORD", "")
    """ES 密码（Basic Auth）。"""

    es_index_prefix: str = _config_str("elasticsearch", "index_prefix", "ES_INDEX_PREFIX", "llm_wiki")
    """ES 索引名前缀。实际索引名为 {prefix}_pages / {prefix}_chunks / {prefix}_graph_edges。"""

    es_analyzer: str = _config_str("elasticsearch", "analyzer", "ES_ANALYZER", "ik_max_word")
    """ES 索引分词器（需安装 ik 插件）。"""

    es_search_analyzer: str = _config_str("elasticsearch", "search_analyzer", "ES_SEARCH_ANALYZER", "ik_smart")
    """ES 搜索分词器。"""

    # ═══════════════════════════════════════════════════════
    # features — 功能开关
    # ═══════════════════════════════════════════════════════
    vector_store_type: str = _config_str("features", "vector_store_type", "VECTOR_STORE_TYPE", "faiss")
    """向量存储后端选择: faiss=FAISS, pgvector=PostgreSQL pgvector。"""

    enable_vector_retrieval: bool = _config_bool("features", "enable_vector_retrieval", "ENABLE_VECTOR_RETRIEVAL", True)
    """是否启用 pgvector 语义检索。同时控制索引时的向量写入和检索时的语义召回。"""

    enable_es_retrieval: bool = _config_bool("features", "enable_es_retrieval", "ENABLE_ES_RETRIEVAL", False)
    """是否启用 ES BM25 词汇检索。同时控制索引时的数据写入和检索时的关键词召回。"""

    enable_local_lexical_retrieval: bool = _config_bool("features", "enable_local_lexical_retrieval", "ENABLE_LOCAL_LEXICAL_RETRIEVAL", True)
    """是否启用本地词汇回退（SQLite FTS5 + 中文分词）。在 ES 和 pgvector 均不可用时作为兜底方案。"""

    # ═══════════════════════════════════════════════════════
    # scenarios — 多场景配置
    # ═══════════════════════════════════════════════════════
    scenarios: dict[str, dict[str, Any]] = _YAML_CONFIG.get("scenarios", {})
    """多场景配置，key 为 source（project_id），value 为覆盖的配置项。"""

    # ═══════════════════════════════════════════════════════
    # indexing — 索引构建参数
    # ═══════════════════════════════════════════════════════
    chunker_version: str = _config_str("indexing", "chunker_version", "CHUNKER_VERSION", "markdown-aware-local-v1")
    """分块器版本标识。版本变化时触发全量重新分块（即使文件内容未变）。"""

    build_embeddings: bool = _config_bool("indexing", "build_embeddings", "BUILD_EMBEDDINGS", False)
    """索引时是否调用 DashScope API 生成向量嵌入。False = 使用 hash 占位向量（仅用于测试）。"""

    # ═══════════════════════════════════════════════════════
    # retrieval — 检索参数
    # ═══════════════════════════════════════════════════════
    default_max_context_size: int = _config_int("retrieval", "default_max_context_size", "DEFAULT_MAX_CONTEXT_SIZE", 204800)
    """构建 LLM 上下文的最大 token 预算（字符数估算）。超出的页面和 chunk 会被截断。"""

    default_top_pages: int = _config_int("retrieval", "default_top_pages", "DEFAULT_TOP_PAGES", 8)
    """最终返回给 LLM 的最大页面数。"""

    default_bm25_top_k: int = _config_int("retrieval", "default_bm25_top_k", "DEFAULT_BM25_TOP_K", 300)
    """ES BM25 / FTS5 召回的最大 chunk 数。"""

    default_vector_top_k: int = _config_int("retrieval", "default_vector_top_k", "DEFAULT_VECTOR_TOP_K", 100)
    """pgvector 语义召回的最大 chunk 数。"""

    default_rerank_top_k: int = _config_int("retrieval", "default_rerank_top_k", "DEFAULT_RERANK_TOP_K", 100)
    """送入重排序模型的候选 chunk 数。"""

    local_lexical_max_chunks: int = _config_int("retrieval", "local_lexical_max_chunks", "LOCAL_LEXICAL_MAX_CHUNKS", 50000)
    """本地词汇回退的 chunk 数量上限。超过此值不执行本地词汇搜索，避免全量内存扫描。"""

    local_vector_max_chunks: int = _config_int("retrieval", "local_vector_max_chunks", "LOCAL_VECTOR_MAX_CHUNKS", 50000)
    """本地暴力向量搜索的 chunk 数量上限。超过此值不执行暴力余弦搜索。"""

    # ═══════════════════════════════════════════════════════
    # logging — 日志
    # ═══════════════════════════════════════════════════════
    log_level: str = _config_str("logging", "level", "RAG_LOG_LEVEL", "INFO")
    """日志级别。可选：DEBUG / INFO / WARNING / ERROR。Loguru 不区分大小写。"""

    log_file: str = _config_str("logging", "file", "RAG_LOG_FILE", "")
    """日志文件路径。空字符串 = 仅输出到 stderr。设置后自动按 10MB 轮转、保留 7 天。"""

    # ═══════════════════════════════════════════════════════
    # agent — Agno RAG Agent（多轮对话 + LLM 调用）
    # ═══════════════════════════════════════════════════════
    enable_agno_agent: bool = _config_bool("agent", "enable_agno_agent", "ENABLE_AGNO_AGENT", True)
    """是否启用 Agno Agent 框架来编排检索和 LLM 调用。"""

    agno_session_db_path: Path = Path(_config_str("agent", "session_db_path", "AGNO_SESSION_DB_PATH", ".rag/agents/sessions.db"))
    """Agno Agent 会话持久化的 SQLite 数据库路径，存储多轮对话历史。"""

    agno_history_runs: int = _config_int("agent", "history_runs", "AGNO_HISTORY_RUNS", 6)
    """多轮对话中保留的历史轮数。每轮包含一次 user 消息和一次 assistant 回复。"""

    # ═══════════════════════════════════════════════════════
    # server — HTTP 服务超时与限制
    # ═══════════════════════════════════════════════════════
    sync_timeout: int = _config_int("server", "sync_timeout", "RAG_SYNC_TIMEOUT", 120)
    """同步端点超时秒数，超时返回 504。"""

    stream_log_interval: int = _config_int("server", "stream_log_interval", "RAG_STREAM_LOG_INTERVAL", 100)
    """流式输出日志每隔多少字符打印一次。"""

    max_request_history: int = _config_int("server", "max_request_history", "RAG_MAX_REQUEST_HISTORY", 20)
    """AnswerRequest.history 中允许的最大 ChatMessage 条数。"""

    max_message_chars: int = _config_int("server", "max_message_chars", "RAG_MAX_MESSAGE_CHARS", 32000)
    """单条 ChatMessage.content 最大字符数。"""

    overload_wait_seconds: float = float(
        _config_value("server", "overload_wait_seconds", "RAG_OVERLOAD_WAIT_SECONDS", 0.05)
    )
    """外部服务并发槽等待时间；超时后快速返回 503。"""

    max_concurrent_chat: int = _config_int(
        "server", "max_concurrent_chat", "RAG_MAX_CONCURRENT_CHAT", 16
    )
    """单进程允许的并发 Chat/流式 Chat 调用数。"""

    max_concurrent_embedding: int = _config_int(
        "server", "max_concurrent_embedding", "RAG_MAX_CONCURRENT_EMBEDDING", 16
    )
    """单进程允许的并发 embedding 调用数。"""

    max_concurrent_rerank: int = _config_int(
        "server", "max_concurrent_rerank", "RAG_MAX_CONCURRENT_RERANK", 8
    )
    """单进程允许的并发 rerank 调用数。"""

    # ═══════════════════════════════════════════════════════
    # session — 会话管理
    # ═══════════════════════════════════════════════════════
    session_max_messages: int = _config_int("session", "max_messages", "RAG_SESSION_MAX_MESSAGES", 20)
    """服务端存储的每会话最大消息数。"""

    history_normalize_max: int = _config_int("session", "normalize_max", "RAG_HISTORY_NORMALIZE_MAX", 12)
    """normalize_history 截断上限。"""

    retrieval_history_limit: int = _config_int("session", "retrieval_history_limit", "RAG_RETRIEVAL_HISTORY_LIMIT", 3)
    """build_retrieval_query 取最近 N 条用户消息。"""

    # ═══════════════════════════════════════════════════════
    # faiss — FAISS 向量存储
    faiss_index_type: str = _config_str("faiss", "index_type", "FAISS_INDEX_TYPE", "FlatIP")
    """FAISS 索引类型：FlatIP=精确内积搜索。"""

    # dashscope — 重试与超时参数
    # ═══════════════════════════════════════════════════════
    dashscope_max_retries: int = _config_int("dashscope", "max_retries", "DASHSCOPE_MAX_RETRIES", 3)
    """DashScope API 最大重试次数（指数退避）。"""

    dashscope_backoff_base: float = float(_config_value("dashscope", "backoff_base", "DASHSCOPE_BACKOFF_BASE", 0.5))
    """重试退避初始等待秒数。"""

    dashscope_request_timeout: int = _config_int("dashscope", "request_timeout", "DASHSCOPE_REQUEST_TIMEOUT", 60)
    """单次 DashScope HTTP 请求超时秒数。"""

    embedding_batch_size: int = _config_int("dashscope", "embedding_batch_size", "EMBEDDING_BATCH_SIZE", 10)
    """批量嵌入时每次 API 调用的最大文本数。"""

    fallback_embedding_dim: int = _config_int("dashscope", "fallback_embedding_dim", "FALLBACK_EMBEDDING_DIM", 384)
    """DashScope 不可用时 hash 嵌入的维度。"""

    # ═══════════════════════════════════════════════════════
    # postgres — 连接池
    # ═══════════════════════════════════════════════════════
    pg_pool_min_size: int = _config_int("postgres", "pool_min_size", "PG_POOL_MIN_SIZE", 2)
    """pgvector 连接池最小连接数。"""

    pg_pool_max_size: int = _config_int("postgres", "pool_max_size", "PG_POOL_MAX_SIZE", 8)
    """pgvector 连接池最大连接数。"""

    # ═══════════════════════════════════════════════════════
    # elasticsearch — 搜索与写入参数
    # ═══════════════════════════════════════════════════════
    es_bulk_batch_size: int = _config_int("elasticsearch", "bulk_batch_size", "ES_BULK_BATCH_SIZE", 500)
    """ES bulk 写入每批最大文档数。"""

    es_search_timeout: int = _config_int("elasticsearch", "search_timeout", "ES_SEARCH_TIMEOUT", 30)
    """ES 搜索单次请求超时秒数。"""

    es_bulk_timeout: int = _config_int("elasticsearch", "bulk_timeout", "ES_BULK_TIMEOUT", 60)
    """ES bulk 写入单次请求超时秒数。"""

    es_title_boost: float = float(_config_value("elasticsearch", "title_boost", "ES_TITLE_BOOST", 3.0))
    """ES 搜索时 title 字段 boost 权重。"""

    es_heading_boost: float = float(_config_value("elasticsearch", "heading_boost", "ES_HEADING_BOOST", 2.0))
    """ES 搜索时 heading_path 字段 boost 权重。"""

    # ═══════════════════════════════════════════════════════
    # indexing — 分块与图构建参数
    # ═══════════════════════════════════════════════════════
    chunk_target_chars: int = _config_int("indexing", "chunk_target_chars", "CHUNK_TARGET_CHARS", 1200)
    """分块目标字符数。"""

    chunk_max_chars: int = _config_int("indexing", "chunk_max_chars", "CHUNK_MAX_CHARS", 1500)
    """分块最大字符数。"""

    chunk_min_chars: int = _config_int("indexing", "chunk_min_chars", "CHUNK_MIN_CHARS", 300)
    """分块最小字符数。"""

    chunk_overlap_chars: int = _config_int("indexing", "chunk_overlap_chars", "CHUNK_OVERLAP_CHARS", 50)
    """相邻块重叠字符数。"""

    chunk_merge_strategy: str = _config_str("indexing", "chunk_merge_strategy", "CHUNK_MERGE_STRATEGY", "auto")
    """分块合并策略: auto=小块合并至target_chars, independent=大块独立不合并。"""

    chunk_standalone_threshold: int = _config_int("indexing", "chunk_standalone_threshold", "CHUNK_STANDALONE_THRESHOLD", 800)
    """independent 策略下，块 ≥ 此长度时独立为一个 chunk，不与其他块合并。"""

    graph_max_shared_source_links: int = _config_int("indexing", "graph_max_shared_source_links", "GRAPH_MAX_SHARED_SOURCE_LINKS", 20)
    """同 source 图边每页采样上限，超此值触发 O(n) 采样。"""

    graph_max_same_type_links: int = _config_int("indexing", "graph_max_same_type_links", "GRAPH_MAX_SAME_TYPE_LINKS", 30)
    """同 type 图边每页采样上限。"""

    # ═══════════════════════════════════════════════════════
    # retrieval — 检索策略调优参数
    # ═══════════════════════════════════════════════════════
    rrf_k: float = float(_config_value("retrieval", "rrf_k", "RRF_K", 60.0))
    """RRF 融合的平滑常数 k，值越大越接近平均融合。"""

    candidate_load_factor: int = _config_int("retrieval", "candidate_load_factor", "CANDIDATE_LOAD_FACTOR", 3)
    """候选加载时 chunk 数 = top_pages × factor。"""

    graph_expand_limit_pages: int = _config_int("retrieval", "graph_expand_limit_pages", "GRAPH_EXPAND_LIMIT_PAGES", 20)
    """图扩展源页面数上限。"""

    graph_expand_per_page: int = _config_int("retrieval", "graph_expand_per_page", "GRAPH_EXPAND_PER_PAGE", 3)
    """图扩展每页最多引入的新页面数。"""

    source_chunk_limit: int = _config_int("retrieval", "source_chunk_limit", "SOURCE_CHUNK_LIMIT", 2)
    """source 类型页面每页最多保留的 anchor chunk 数。"""

    anchor_chunks_per_page: int = _config_int("retrieval", "anchor_chunks_per_page", "ANCHOR_CHUNKS_PER_PAGE", 3)
    """页面聚合时每页取的 anchor chunk 数。"""

    scored_chunks_per_page: int = _config_int("retrieval", "scored_chunks_per_page", "SCORED_CHUNKS_PER_PAGE", 3)
    """候选评分后每页保留的 chunk 数。"""

    # ── 候选评分权重 ──
    score_weight_page_rrf: float = float(_config_value("retrieval", "score_weight_page_rrf", "SCORE_WEIGHT_PAGE_RRF", 0.25))
    score_weight_chunk_rrf: float = float(_config_value("retrieval", "score_weight_chunk_rrf", "SCORE_WEIGHT_CHUNK_RRF", 0.35))
    score_weight_rerank: float = float(_config_value("retrieval", "score_weight_rerank", "SCORE_WEIGHT_RERANK", 0.20))
    score_weight_title_match: float = float(_config_value("retrieval", "score_weight_title_match", "SCORE_WEIGHT_TITLE_MATCH", 0.10))
    score_weight_graph: float = float(_config_value("retrieval", "score_weight_graph", "SCORE_WEIGHT_GRAPH", 0.10))
    score_weight_type: float = float(_config_value("retrieval", "score_weight_type", "SCORE_WEIGHT_TYPE", 0.05))

    # ── 页面聚合类型 boost ──
    page_type_entity_boost: float = float(_config_value("retrieval", "page_type_entity_boost", "PAGE_TYPE_ENTITY_BOOST", 1.18))
    page_type_concept_boost: float = float(_config_value("retrieval", "page_type_concept_boost", "PAGE_TYPE_CONCEPT_BOOST", 1.08))
    page_type_source_boost: float = float(_config_value("retrieval", "page_type_source_boost", "PAGE_TYPE_SOURCE_BOOST", 0.72))
    page_title_boost: float = float(_config_value("retrieval", "page_title_boost", "PAGE_TITLE_BOOST", 0.005))
    page_tail_weight: float = float(_config_value("retrieval", "page_tail_weight", "PAGE_TAIL_WEIGHT", 0.08))

    # ── 词汇回退评分 ──
    lexical_exact_title_score: float = float(_config_value("retrieval", "lexical_exact_title_score", "LEXICAL_EXACT_TITLE_SCORE", 250.0))
    lexical_phrase_title_score: float = float(_config_value("retrieval", "lexical_phrase_title_score", "LEXICAL_PHRASE_TITLE_SCORE", 80.0))
    lexical_phrase_heading_score: float = float(_config_value("retrieval", "lexical_phrase_heading_score", "LEXICAL_PHRASE_HEADING_SCORE", 45.0))
    lexical_phrase_content_score: float = float(_config_value("retrieval", "lexical_phrase_content_score", "LEXICAL_PHRASE_CONTENT_SCORE", 20.0))
    lexical_phrase_content_limit: int = _config_int("retrieval", "lexical_phrase_content_limit", "LEXICAL_PHRASE_CONTENT_LIMIT", 8)
    lexical_token_title_score: float = float(_config_value("retrieval", "lexical_token_title_score", "LEXICAL_TOKEN_TITLE_SCORE", 8.0))
    lexical_token_heading_score: float = float(_config_value("retrieval", "lexical_token_heading_score", "LEXICAL_TOKEN_HEADING_SCORE", 5.0))
    lexical_token_content_score: float = float(_config_value("retrieval", "lexical_token_content_score", "LEXICAL_TOKEN_CONTENT_SCORE", 1.0))
    lexical_chunk_type_entity_boost: float = float(_config_value("retrieval", "lexical_chunk_type_entity_boost", "LEXICAL_CHUNK_TYPE_ENTITY_BOOST", 1.18))
    lexical_chunk_type_concept_boost: float = float(_config_value("retrieval", "lexical_chunk_type_concept_boost", "LEXICAL_CHUNK_TYPE_CONCEPT_BOOST", 1.1))
    lexical_chunk_type_source_boost: float = float(_config_value("retrieval", "lexical_chunk_type_source_boost", "LEXICAL_CHUNK_TYPE_SOURCE_BOOST", 0.82))

    # ═══════════════════════════════════════════════════════
    # context — 上下文构建参数
    # ═══════════════════════════════════════════════════════
    index_budget_fraction: float = float(_config_value("context", "index_budget_fraction", "CTX_INDEX_BUDGET_FRAC", 0.05))
    """页面索引列表占上下文字符预算的比例。"""

    page_budget_fraction: float = float(_config_value("context", "page_budget_fraction", "CTX_PAGE_BUDGET_FRAC", 0.5))
    """页面内容总共占上下文字符预算的比例。"""

    per_page_budget_fraction: float = float(_config_value("context", "per_page_budget_fraction", "CTX_PER_PAGE_FRAC", 0.3))
    """单页最多占页面预算的比例。"""

    per_page_budget_floor: int = _config_int("context", "per_page_budget_floor", "CTX_PER_PAGE_FLOOR", 5000)
    """单页字符预算下限。"""

    section_first_threshold: int = _config_int("context", "section_first_threshold", "CTX_SECTION_FIRST_THRESHOLD", 8000)
    """页面内容长度超过此值触发 section 截断策略。"""

    section_neighbor_chars: int = _config_int("context", "section_neighbor_chars", "CTX_SECTION_NEIGHBOR_CHARS", 1000)
    """section 窗口向上下扩展的字符数。"""

    page_prefix_chars: int = _config_int("context", "page_prefix_chars", "CTX_PAGE_PREFIX_CHARS", 1200)
    """页面内容前缀截断长度（section 截断策略的回退）。"""

    section_search_fragment_chars: int = _config_int("context", "section_search_fragment_chars", "CTX_SECTION_FRAGMENT_CHARS", 200)
    """section 搜索时用于匹配的最大片段长度。"""

    # ═══════════════════════════════════════════════════════
    # sqlite — 本地存储参数
    # ═══════════════════════════════════════════════════════
    sqlite_busy_timeout: int = _config_int("sqlite", "busy_timeout", "SQLITE_BUSY_TIMEOUT", 5000)
    """SQLite 并发忙等待超时毫秒数。"""

    sqlite_in_batch_size: int = _config_int("sqlite", "in_batch_size", "SQLITE_IN_BATCH_SIZE", 900)
    """SQL IN 查询分批大小（SQLite 变量数上限 ~999）。"""

    fts_max_query_terms: int = _config_int("sqlite", "fts_max_query_terms", "FTS_MAX_QUERY_TERMS", 64)
    """FTS5 查询最大分词数。"""

    # ═══════════════════════════════════════════════════════
    # tokenizer — 分词与估算
    # ═══════════════════════════════════════════════════════
    token_estimate_cjk_ratio: float = float(_config_value("tokenizer", "cjk_ratio", "TOKEN_CJK_RATIO", 1.5))
    """中文 token 估算：每个 CJK 字符平均对应多少 token。"""

    token_estimate_ascii_ratio: float = float(_config_value("tokenizer", "ascii_ratio", "TOKEN_ASCII_RATIO", 4.0))
    """英文 token 估算：每个 ASCII 字符平均对应多少 token。"""

    # ═══════════════════════════════════════════════════════
    # prompts — LLM Prompt 模板（支持热更新，不需重启）
    # ═══════════════════════════════════════════════════════
    system_prompt: str = _config_str(
        "prompts", "system",
        "RAG_SYSTEM_PROMPT",
        "You are a knowledgeable wiki assistant.\n\n"
        "## Rules\n"
        "- Answer concisely based on the wiki pages below. No lengthy explanations.\n"
        "- If information is missing, say so in one sentence.\n"
        "- Cite sources with [page number] only when needed, e.g. [1].\n\n"
        "## MANDATORY OUTPUT LANGUAGE: Chinese\n"
        "Write every response entirely in Chinese.\n",
    )
    """RAG 问答 System Prompt。"""

    rewrite_timeout: int = _config_int("prompts", "rewrite_timeout", "REWRITE_TIMEOUT", 10)

    rewrite_model: str = _config_str("prompts", "rewrite_model", "REWRITE_MODEL", "qwen-turbo")

    query_rewrite_system_prompt: str = _config_str(
        "prompts", "query_rewrite_system",
        "RAG_QUERY_REWRITE_SYSTEM_PROMPT",
        "判断用户当前问题是否需要结合对话历史改写为独立检索查询，输出JSON。\n"
        "\n"
        "【判断标准】\n"
        "需要改写(rewrite=true)：问题含代词、省略主语、或无上下文无法独立理解\n"
        "不需改写(rewrite=false)：问题已包含完整实体名且语义独立\n"
        "\n"
        '输出格式：{"rewrite": true, "query": "<改写后>"} 或 {"rewrite": false}\n'
        "只输出JSON，不要其他内容。",
    )
    """多轮对话 Query 改写的 System Prompt。"""

    query_rewrite_user_template: str = _config_str(
        "prompts", "query_rewrite_user_template",
        "RAG_QUERY_REWRITE_USER_TEMPLATE",
        "对话历史：\n{history}\n\n用户当前问题：{query}\n\n改写后的独立查询：",
    )
    """Query 改写 User Prompt 模板。{history} 和 {query} 为运行时替换变量。"""

    user_prompt_inline_instruction: str = _config_str(
        "prompts", "user_inline_instruction",
        "RAG_USER_INLINE_INSTRUCTION",
        "直接回答问题，简洁为主，必要时引用来源编号。",
    )
    """build_user_prompt 末尾追加的行为指令。"""

    # ═══════════════════════════════════════════════════════
    # 计算属性 — 由上述字段组合推导
    # ═══════════════════════════════════════════════════════

    @property
    def dashscope_enabled(self) -> bool:
        """DashScope API 是否可用（已配置 API Key）。"""
        return bool(self.dashscope_api_key)

    @property
    def pgvector_enabled(self) -> bool:
        """pgvector 是否可用（已配置主机地址 + 向量检索开关已开启）。"""
        return bool(self.pg_host) and self.enable_vector_retrieval

    @property
    def es_enabled(self) -> bool:
        """ES 是否可用（已配置 URL）。"""
        return bool(self.es_url)

    @property
    def es_retrieval_enabled(self) -> bool:
        """ES BM25 检索是否可用（已配置 URL + 开关已开启）。"""
        return self.es_enabled and self.enable_es_retrieval

    @property
    def es_indexing_enabled(self) -> bool:
        """ES 索引写入是否可用（与 es_retrieval_enabled 相同，因为写入和检索共用一个开关）。"""
        return self.es_enabled and self.enable_es_retrieval

    @property
    def es_pages_index(self) -> str:
        """ES 页面索引名 = {prefix}_pages。"""
        return f"{self.es_index_prefix}_pages"

    @property
    def es_chunks_index(self) -> str:
        """ES 分块索引名 = {prefix}_chunks。"""
        return f"{self.es_index_prefix}_chunks"

    @property
    def es_graph_edges_index(self) -> str:
        """ES 图边索引名 = {prefix}_graph_edges。"""
        return f"{self.es_index_prefix}_graph_edges"

    @property
    def faiss_enabled(self) -> bool:
        return self.vector_store_type == "faiss" and self.enable_vector_retrieval

    @property
    def pgvector_enabled(self) -> bool:
        return self.vector_store_type == "pgvector" and bool(self.pg_host) and self.enable_vector_retrieval

    @property
    def vector_store_enabled(self) -> bool:
        if self.vector_store_type == "pgvector":
            return self.pgvector_enabled
        return self.faiss_enabled

    @property
    def pg_dsn(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

# 全局单例
settings = Settings()
_settings_load_time = os.environ.get("ENV", "dev"), str(_YAML_PATH)


def reload_settings() -> dict[str, object]:
    """热重载配置，返回变更信息。"""
    global _YAML_CONFIG, settings, _settings_load_time
    _YAML_CONFIG = _load_yaml_config(_YAML_PATH)
    old = settings
    settings = Settings()
    _settings_load_time = os.environ.get("ENV", "dev"), str(_YAML_PATH)
    return {
        "env": _settings_load_time[0],
        "config_path": _settings_load_time[1],
        "changed_keys": [
            field.name for field in Settings.__dataclass_fields__.values()
            if getattr(old, field.name) != getattr(settings, field.name)
        ],
    }
