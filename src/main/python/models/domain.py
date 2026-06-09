# ============================================================
# 领域模型（dataclass）
# ============================================================
# Page:  一个 Markdown 文件解析后的完整页面
# Chunk: 页面的分块，是检索的最小单位
# GraphEdge: 页面之间的关联边（wikilink / 共享源 / 同类型）
# JobStatus: 异步索引作业的状态
#
# 注意：Pydantic 请求/响应模型在 schemas.py 中
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    """知识库页面。

    由 Markdown 文件解析而来，包含正文全文、元数据、索引信息。
    type 字段取值: entity / concept / source
    """
    project_id: str          # 所属项目
    page_id: str             # 唯一标识（stable_id 或 dedup_key）
    path: str               # 相对路径
    title: str              # 标题（h1 或文件名）
    type: str               # 页面类型
    sources: list[str]       # 来源列表
    wikilinks: list[str]     # [[wikilink]] 引用列表
    content: str            # 正文全文
    metadata: dict[str, Any] # frontmatter 元数据
    content_sha256: str     # 内容哈希（增量检测用）
    mtime: float            # 文件修改时间
    chunk_count: int = 0    # 分块数量
    indexed_at: str = ""    # 索引时间戳


@dataclass
class Chunk:
    """页面的一个分块，是检索的最小粒度。

    相邻块指针（prev_chunk_id / next_chunk_id）用于相邻块扩展检索。
    vector 由 embed_text() 生成，空列表表示未嵌入。
    """
    project_id: str
    page_id: str            # 所属页面
    chunk_id: str           # 唯一标识（sha256 前16位 hex）
    path: str               # 源文件路径
    title: str              # 所属页面标题
    heading_path: str       # 标题层级路径（如 "产品 > 特色体检 > 流程"）
    type: str               # 继承自 Page
    sources: list[str]       # 继承自 Page
    content: str            # 块文本
    chunk_index: int        # 块在页面中的序号
    prev_chunk_id: str | None = None  # 前一个块的 ID
    next_chunk_id: str | None = None  # 后一个块的 ID
    token_estimate: int = 0 # token 估算（content 字符数 / 4）
    vector: list[float] = field(default_factory=list)  # 嵌入向量


@dataclass
class GraphEdge:
    """页面之间的关联边。

    边类型:
      - wikilink: 页面 A 引用了页面 B 的 [[wikilink]]（权重 3.0）
      - shared_source: 两个页面共享同一来源（权重 4.0）
      - same_type: 两个页面类型相同（权重 1.0）
    """
    project_id: str
    source_page_id: str     # 源页面
    target_page_id: str     # 目标页面
    edge_type: str          # wikilink / shared_source / same_type
    weight: float           # 权重，用于图扩展时的分数计算


@dataclass
class JobStatus:
    """异步索引作业状态。"""
    job_id: str
    status: str             # running / completed / failed
    pages_total: int = 0
    pages_done: int = 0
    chunks_indexed: int = 0
    embeddings_done: int = 0
    duration_ms: float = 0.0
    error: str | None = None
