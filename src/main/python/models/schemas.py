# ============================================================
# API 请求/响应模型（Pydantic）
# ============================================================
# IndexRequest     → POST /rag/index/project
# DataProcessRequest → POST /data/process
# ContextRequest   → POST /rag/context, /rag/search/debug
# AnswerRequest    → POST /rag/answer (继承 ContextRequest)
# ChatMessage      → 多轮对话消息 {role, content}
# ============================================================

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.main.python.config import settings


class IndexRequest(BaseModel):
    """索引请求。

    POST /rag/index/project
    """
    project_id: str                                     # 项目唯一标识
    project_path: str = "data"                          # Markdown 文件根目录
    force: bool = False                                 # True = 全量重建索引
    build_embeddings: bool = settings.build_embeddings   # 是否生成向量嵌入


class DataProcessRequest(BaseModel):
    """数据处理请求（IndexRequest 的快捷方式，带默认 project_id）。

    POST /data/process
    """
    project_id: str = "pingan-zhenxiang"
    data_path: str = str(settings.default_data_path)
    force: bool = True
    build_embeddings: bool = settings.build_embeddings


class ContextRequest(BaseModel):
    """检索上下文请求。

    POST /rag/context 和 POST /rag/search/debug 共用。
    可通过 include_es / include_vector 动态开关检索通路（None = 使用配置默认值）。
    """
    project_id: str
    query: str                                          # 用户查询文本
    max_context_size: int = Field(default=settings.default_max_context_size, ge=100)  # 上下文 token 预算
    top_pages: int = Field(default=settings.default_top_pages, ge=1)                   # 返回最多 N 个页面
    bm25_top_k: int = Field(default=settings.default_bm25_top_k, ge=1)                 # ES BM25 召回数
    vector_top_k: int = Field(default=settings.default_vector_top_k, ge=1)             # pgvector 召回数
    rerank_top_k: int = Field(default=settings.default_rerank_top_k, ge=1)             # 重排序候选数
    include_es: Optional[bool] = None                   # 覆盖 ES BM25 开关
    include_vector: Optional[bool] = None               # 覆盖向量检索开关
    include_lexical: Optional[bool] = None              # 覆盖本地词汇检索开关
    include_graph: bool = True                          # 是否启用图扩展
    include_neighbor_chunks: bool = True                # 是否启用相邻块扩展


class ChatMessage(BaseModel):
    """多轮对话消息。

    role: "user" 或 "assistant"
    """
    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=0, max_length=settings.max_message_chars)


class AnswerRequest(ContextRequest):
    """RAG 问答请求（继承 ContextRequest 的所有检索参数）。

    POST /rag/answer
    """
    session_id: Optional[str] = Field(default=None, max_length=128)  # 会话 ID（多轮对话用）
    user_id: Optional[str] = Field(default=None, max_length=128)     # 用户 ID
    history: list[ChatMessage] = Field(default_factory=list, max_length=settings.max_request_history)  # 历史消息列表


class AnswerResponse(BaseModel):
    """RAG 问答响应模型。

    POST /rag/answer 和 POST /rag/answer/stream (done 事件) 共用。
    """
    answer: str = ""
    end_flag: int = Field(default=1, description="流式结束标志: 0=未结束, 1=已结束")
    total_ms: float = Field(default=0, description="请求总耗时（毫秒）")
    retrieval_ms: float = Field(default=0, description="检索阶段耗时（毫秒）")
    llm_ms: float = Field(default=0, description="LLM 生成耗时（毫秒）")
    llm_error: Optional[str] = None
    context: Optional[dict[str, object]] = None
    session_id: Optional[str] = None
    agent_engine: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# AskBob 协议模型 — /api/v1/chat/askbob
# ═══════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """AskBob 协议请求模型。"""
    session_id: str = Field(default="", max_length=128)
    user_text: str = Field(..., min_length=1, description="用户问题")
    user_action: str = Field(default="write")
    action_scenario: str = Field(default="")
    trace_id: str = Field(default="", max_length=128)
    user_id: str = Field(default="", max_length=128)
    ts: str = Field(default="")
    token: str = Field(default="")
    source: str = Field(default="askbob", description="对应 project_id")
    extra_input_params: dict[str, object] = Field(default_factory=dict)


class ChatStatus(BaseModel):
    """Chat 响应状态。"""
    report_ready_flag: int = 0
    return_task_flag: int = 1


class ChatExtraOutput(BaseModel):
    """Chat 响应额外数据。"""
    query: str = ""
    rewritten_query: str = ""
    first_frame_time: float = 0.0
    final_frame_time: float = 0.0
    pages: list[dict[str, object]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class ChatData(BaseModel):
    """Chat 响应 data 段。"""
    robot_text: str = ""
    source: str = ""
    end_flag: int = 1
    status: ChatStatus = Field(default_factory=ChatStatus)
    extra_output_params: ChatExtraOutput = Field(default_factory=ChatExtraOutput)


class ChatResponse(BaseModel):
    """Chat 响应（AskBob 协议）。"""
    code: int = 0
    data: ChatData = Field(default_factory=ChatData)
