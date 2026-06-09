# LLM RAG V3 API 文档

> 基础地址: `http://localhost:8010`

## 快速参考

```bash
# 健康检查
curl http://localhost:8010/health

# 索引项目
curl -X POST http://localhost:8010/rag/index/project \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"my-project","project_path":"data","force":true}'

# 检索上下文
curl -X POST http://localhost:8010/rag/context \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"my-project","query":"家庭医生服务有哪些内容？","top_pages":10}'

# 同步问答（原协议）
curl -X POST http://localhost:8010/rag/answer \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"my-project","query":"家庭医生服务有哪些内容？"}'

# 流式问答（原协议）
curl -N -X POST http://localhost:8010/rag/answer/stream \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"my-project","query":"家庭医生服务有哪些内容？"}'

# 同步问答（AskBob 协议）
curl -X POST http://localhost:8010/rag/chat \
  -H 'Content-Type: application/json' \
  -d '{"source":"my-project","user_text":"50岁老人买什么保险合适","session_id":"demo-001"}'

# 流式问答（AskBob 协议）
curl -N -X POST http://localhost:8010/rag/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"source":"my-project","user_text":"50岁老人买什么保险合适"}'

# 会话管理
curl http://localhost:8010/sessions/demo-001
curl -X DELETE http://localhost:8010/sessions/demo-001
```

## 目录

- [1. 服务信息](#1-服务信息)
- [2. 健康检查](#2-健康检查)
- [3. 深度就绪检查](#3-深度就绪检查)
- [4. 索引项目](#4-索引项目)
- [5. 处理数据](#5-处理数据)
- [6. 查询作业状态](#6-查询作业状态)
- [7. 检索上下文](#7-检索上下文)
- [8. 调试检索](#8-调试检索)
- [9. 同步问答（原协议）](#9-同步问答（原协议）)
- [10. 流式问答（原协议）](#10-流式问答（原协议）)
- [11. 会话管理](#11-会话管理)
- [12. 同步问答（AskBob 协议）](#12-同步问答（askbob-协议）)
- [13. 流式问答（AskBob 协议）](#13-流式问答（askbob-协议）)
- [14. 热重载配置](#14-热重载配置)
- [附：检索参数说明](#附检索参数说明)

---

## 1. 服务信息

```
GET /
```

**响应：**

```json
{
  "service": "LLM RAG V3",
  "version": "0.2.0",
  "status": "running"
}
```

---

## 2. 健康检查

```
GET /health
```

**入参：** 无

---

**响应：**

```json
{
  "status": "ok",
  "storage_dir": ".rag",
  "elasticsearch": {
    "configured": true,
    "indexing_enabled": true,
    "retrieval_enabled": true,
    "index_prefix": "llm_wiki"
  },
  "retrieval": {
    "vector_enabled": true,
    "pgvector_enabled": true,
    "local_lexical_enabled": true,
    "default_max_context_size": 204800,
    "default_top_pages": 8,
    "default_bm25_top_k": 300,
    "default_vector_top_k": 100,
    "default_rerank_top_k": 100
  },
  "indexing": { "build_embeddings": false },
  "logging": { "level": "DEBUG", "file": "logs/dev.log" },
  "sqlite": { "fts5_available": true },
  "dashscope": {
    "configured": true,
    "embedding_model": "text-embedding-v4",
    "rerank_model": "qwen3-rerank",
    "llm_model": "qwen3.6-35b-a3b"
  }
}
```

---

## 3. 深度就绪检查

```
GET /health/ready
```

**入参：** 无

**响应：**

```json
{
  "status": "ready",
  "components": {
    "indexes": {
      "ready": true,
      "count": 3,
      "projects": ["wiki", "pingan-zhenxiang", "demo"]
    },
    "sessions": {
      "active": 12,
      "ready": true
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `status` | `"ready"` = 全正常，`"degraded"` = 部分组件异常 |
| `components.indexes.ready` | 索引数据是否就绪 |
| `components.indexes.count` | 已索引项目数 |
| `components.sessions.active` | 活跃会话数 |

---

## 4. 索引项目

```
POST /rag/index/project
```

扫描 Markdown 文件，解析、分块、嵌入，写入 SQLite + pgvector + Elasticsearch。

**请求：**

```json
{
  "project_id": "my-project",       // 必填，项目唯一标识
  "project_path": "data",           // Markdown 文件根目录，默认 "data"
  "force": false,                   // 是否全量重建，默认 false（增量）
  "build_embeddings": true          // 是否生成向量嵌入
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `project_id` | string | 是 | - | 项目唯一标识 |
| `project_path` | string | 否 | `"data"` | Markdown 文件根目录 |
| `force` | bool | 否 | `false` | `true` = 全量重建，`false` = 增量（对比 manifest 只处理变化文件） |
| `build_embeddings` | bool | 否 | 配置值 | `true` = 调用 DashScope 生成向量，`false` = hash 占位 |

**响应：**

```json
{
  "project_id": "my-project",
  "pages_indexed": 27,
  "chunks_indexed": 59,
  "edges_indexed": 593,
  "files_reused": 20,
  "files_reindexed": 7,
  "files_deleted": 0,
  "pgvector_indexed": true,
  "pgvector_error": null,
  "elasticsearch_indexed": true,
  "elasticsearch_error": null,
  "index_path": ".rag/indexes/my-project.json",
  "manifest_path": ".rag/manifests/my-project.json"
}
```

---

## 5. 处理数据

```
POST /data/process
```

索引项目的快捷方式，默认 `project_id = "pingan-zhenxiang"`、`force = true`。

**请求：**

```json
{
  "project_id": "pingan-zhenxiang",
  "data_path": "data",
  "force": true,
  "build_embeddings": true
}
```

参数同 [索引项目](#4-索引项目)。

---

## 6. 查询作业状态

```
GET /jobs/{job_id}
```

索引是异步作业，返回作业状态和进度。

**响应：**

```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "completed",       // running | completed | failed
  "pages_total": 27,
  "pages_done": 27,
  "chunks_indexed": 59,
  "embeddings_done": 59,
  "duration_ms": 36542.1,
  "error": null
}
```

---

## 7. 检索上下文

```
POST /rag/context
```

仅检索，不调用 LLM。返回相关页面及其分块内容。

**请求：**

```json
{
  "project_id": "pingan-zhenxiang",
  "query": "家庭医生服务有哪些内容？",
  "max_context_size": 204800,
  "top_pages": 8,
  "bm25_top_k": 300,
  "vector_top_k": 100,
  "rerank_top_k": 100,
  "include_es": true,
  "include_vector": true,
  "include_lexical": true,
  "include_graph": true,
  "include_neighbor_chunks": true
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `project_id` | string | 是 | - | 项目标识 |
| `query` | string | 是 | - | 检索查询文本 |
| `max_context_size` | int | 否 | 204800 | 上下文 token 预算（字符数） |
| `top_pages` | int | 否 | 8 | 返回的最大页面数 |
| `bm25_top_k` | int | 否 | 300 | ES BM25 召回数 |
| `vector_top_k` | int | 否 | 100 | pgvector 召回数 |
| `rerank_top_k` | int | 否 | 100 | 重排序候选数 |
| `include_es` | bool/null | 否 | null | 是否启用 ES BM25 召回（null=全局配置） |
| `include_vector` | bool/null | 否 | null | 是否启用 pgvector 语义召回 |
| `include_lexical` | bool/null | 否 | null | 是否启用 SQLite FTS5 词汇召回 |
| `include_graph` | bool | 否 | true | 是否启用图扩展 |
| `include_neighbor_chunks` | bool | 否 | true | 是否扩展相邻块 |

**响应：**

```json
{
  "pages": [
    {
      "number": 1,
      "page_id": "family-doctor-service",
      "title": "家庭医生服务",
      "path": "entities/家庭医生服务.md",
      "score": 0.8667,
      "source": "hybrid",
      "chunks": [
        {
          "chunk_id": "family-doctor-service#0000",
          "heading_path": "家庭医生服务 > 概念层级",
          "content": "# 家庭医生服务\n\n## 服务定义...",
          "score": 0.8667
        }
      ]
    }
  ],
  "page_list": "1. 家庭医生服务  (entities/家庭医生服务.md)\n2. ...",
  "index": "## 索引\n- 家庭医生服务 → page 1...",
  "pages_context": "## 1. 家庭医生服务\n\n# 家庭医生服务...",
  "metrics": {
    "bm25_latency_ms": 50.5,
    "vector_latency_ms": 567.6,
    "fts5_latency_ms": 3.8,
    "rrf_latency_ms": 0.02,
    "graph_latency_ms": 9.8,
    "scoring_latency_ms": 720.1,
    "context_build_latency_ms": 0.07
  },
  "retrieval_mode": "hybrid",
  "fallback_reasons": [
    "bm25: 0 results",
    "pgvector: 10 results",
    "fts5: 10 results"
  ]
}
```

---

## 8. 调试检索

```
POST /rag/search/debug
```

与 `/rag/context` 参数完全相同，但返回额外的 `debug` 字段，包含每路召回的原始结果。

**请求：** 同 [检索上下文](#7-检索上下文)。

**响应：** 在 `/rag/context` 基础上增加：

```json
{
  "...": "...",
  "debug": {
    "bm25": [["chunk_id_1", 9.5], ["chunk_id_2", 8.2]],
    "vector": [["chunk_id_3", 0.85], ["chunk_id_4", 0.73]],
    "fts5": [["chunk_id_5", 12.6], ["chunk_id_6", 10.1]],
    "rrf": [{"id": "chunk_1", "score": 0.032}, ...],
    "pages": [{"page_id": "...", "score": 0.86}, ...],
    "graph_expansions": [{"page_id": "...", "graph_score": 3.0}, ...],
    "selected": [{"page_id": "...", "chunk_ids": [...], "score": 0.86}]
  }
}
```

---

## 9. 同步问答（原协议）

```
POST /rag/answer
```

检索 + LLM 生成答案，等待完整响应后返回。

**请求：** 继承 [检索上下文](#7-检索上下文) 全部参数，增加：

```json
{
  "project_id": "pingan-zhenxiang",
  "query": "家庭医生服务有哪些内容？",
  "session_id": "user-session-001",
  "user_id": "user-123",
  "history": [
    { "role": "user", "content": "你好" },
    { "role": "assistant", "content": "你好！请问有什么可以帮助你的？" }
  ]
}
```

| 额外参数 | 类型 | 必填 | 默认值 | 说明 |
|----------|------|------|--------|------|
| `session_id` | string | 否 | null | 会话 ID，用于多轮对话记忆 |
| `user_id` | string | 否 | null | 用户 ID |
| `history` | ChatMessage[] | 否 | [] | 历史对话消息 |

**响应：**

```json
{
  "answer": "家庭医生服务包括：\n\n1. **专属签约**...\n\n<!-- cited: 1, 3, 5 -->",
  "end_flag": 1,
  "total_ms": 1523.0,
  "retrieval_ms": 345.0,
  "llm_ms": 1178.0,
  "llm_error": null,
  "context": {
    "pages": [...],
    "metrics": {...},
    "fallback_reasons": [...]
  },
  "session_id": "user-session-001",
  "agent_engine": "agno"
}
```

| 字段 | 说明 |
|------|------|
| `answer` | LLM 生成的答案（Markdown 格式，含 `[[wikilink]]` 和 `[页码]` 引用） |
| `end_flag` | 流式结束标志，同步接口始终为 `1`（已结束）。`0`=未结束，`1`=已结束 |
| `total_ms` | 请求总耗时（毫秒），含检索 + LLM 生成 |
| `retrieval_ms` | 检索阶段耗时（毫秒） |
| `llm_ms` | LLM 生成阶段耗时（毫秒） |
| `llm_error` | LLM 错误信息（null = 正常） |
| `context` | 检索上下文（同 /rag/context 响应） |
| `session_id` | 会话 ID |
| `agent_engine` | 使用的 Agent 引擎（`agno` 或 `dashscope_fallback`） |

---

## 10. 流式问答（原协议）

```
POST /rag/answer/stream
```

参数同 [同步问答](#9-同步问答（原协议）)，但以 Server-Sent Events (SSE) 格式流式返回。

**请求：** 同 `/rag/answer`。

**SSE 事件流：**

每个事件的 JSON 结构与 [同步问答](#9-同步问答（原协议）) 响应完全一致，仅 `answer` 逐块累加、`end_flag` 和耗时字段动态更新。

```
data: {"answer":"根据提供的","end_flag":0,"total_ms":380,"retrieval_ms":345,"llm_ms":35,"llm_error":null,"context":{...},"session_id":null,"agent_engine":"dashscope"}

data: {"answer":"根据提供的资料，**家庭医生","end_flag":0,"total_ms":412,"retrieval_ms":345,"llm_ms":67,"llm_error":null,"context":{...},"session_id":null,"agent_engine":"dashscope"}

data: {"answer":"根据提供的资料，**家庭医生**是围绕个人及家庭健康...","end_flag":0,"total_ms":450,"retrieval_ms":345,"llm_ms":105,"llm_error":null,"context":{...},"session_id":null,"agent_engine":"dashscope"}

data: {"answer":"根据提供的资料，**家庭医生**是围绕个人及家庭健康...（完整答案）","end_flag":1,"total_ms":1856,"retrieval_ms":345,"llm_ms":1511,"llm_error":null,"context":{...},"session_id":null,"agent_engine":"dashscope"}
```

**字段说明：** 同 [同步问答响应字段](#9-同步问答（原协议）)，额外说明：

| 字段 | 流式行为 |
|------|----------|
| `answer` | 从开始到当前的全部累加文本，逐块增长 |
| `end_flag` | `0`=传输中，`1`=已结束 |
| `total_ms` | 实时更新，从请求进入到当前时刻的总耗时 |
| `llm_ms` | 实时更新，从 LLM 开始生成到当前时刻的耗时 |
| `retrieval_ms` | 固定值，检索阶段耗时（仅检索一次） |
| `context` | 固定值，与同步接口结构一致 |

**curl 示例：**

```bash
curl -N -X POST http://localhost:8010/rag/answer/stream \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"pingan-zhenxiang","query":"家庭医生服务有哪些"}'
```

**JavaScript 前端示例：**

```javascript
const response = await fetch('/rag/answer/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ project_id: 'pingan-zhenxiang', query: '...' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6));

    // event 结构与 /rag/answer 同步响应完全一致
    // answer 是当前累加的全部文本，直接替换显示即可
    outputEl.textContent = event.answer;

    if (event.end_flag === 1) {
      console.log('回答完成', `检索: ${event.retrieval_ms}ms`, `LLM: ${event.llm_ms}ms`, `总耗时: ${event.total_ms}ms`);
    }
  }
}
```

---

## 11. 会话管理

### 获取会话

```
GET /sessions/{session_id}
```

**响应：**

```json
{
  "session_id": "user-session-001",
  "history": [
    { "role": "user", "content": "家庭医生有哪些内容？" },
    { "role": "assistant", "content": "家庭医生服务包括..." }
  ]
}
```

### 清除会话

```
DELETE /sessions/{session_id}
```

**响应：**

```json
{
  "session_id": "user-session-001",
  "cleared": true
}
```

---

## 12. 同步问答（AskBob 协议）

```
POST /rag/chat
```

AskBob 协议同步问答。采用 `{ code, data }` 响应格式，`source` 映射到内部 `project_id`，`user_text` 映射到 `query`。

**请求：**

```json
{
  "session_id": "20250214#3e27e75f376c474397648fb0d55c085d",
  "user_text": "50岁老人买什么保险合适",
  "user_action": "write",
  "action_scenario": "",
  "trace_id": "20250214#dc8e665fffa54d2b8603fe78978625f4",
  "user_id": "TEST0237",
  "ts": "1739514036376",
  "token": "78b534a95a8e06f0fd147aa983710300",
  "source": "wiki",
  "extra_input_params": {}
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_text` | string | 是 | 用户问题，映射到内部 `query` |
| `source` | string | 否 | 项目标识，映射到内部 `project_id`，默认 `"askbob"` |
| `session_id` | string | 否 | 会话 ID，用于多轮对话 |
| `user_id` | string | 否 | 用户 ID |
| `trace_id` | string | 否 | 请求追踪 ID |
| `ts` | string | 否 | 请求时间戳 |
| `token` | string | 否 | 认证 token（预留） |
| `user_action` | string | 否 | 用户动作，默认 `"write"` |
| `action_scenario` | string | 否 | 场景标识 |
| `extra_input_params` | dict | 否 | 扩展参数 |

**响应：**

```json
{
  "code": 0,
  "data": {
    "robot_text": "根据资料，50岁老人建议考虑...",
    "source": "wiki",
    "end_flag": 1,
    "status": {
      "report_ready_flag": 0,
      "return_task_flag": 1
    },
    "extra_output_params": {
      "query": "50岁老人买什么保险合适",
      "rewritten_query": "",
      "first_frame_time": 345.1,
      "final_frame_time": 1523.5,
      "pages": [
        {
          "number": 1,
          "page_id": "...",
          "title": "...",
          "path": "...",
          "score": 0.8667,
          "source": "hybrid",
          "chunks": [
            {
              "chunk_id": "...",
              "heading_path": "...",
              "content": "...",
              "score": 0.8667
            }
          ]
        }
      ],
      "metrics": {
        "bm25_latency_ms": 0.006,
        "vector_latency_ms": 553.2,
        "fts5_latency_ms": 38.9,
        "scoring_latency_ms": 1724.4,
        "context_build_latency_ms": 0.35
      }
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `code` | 状态码，`0` = 成功，非 `0` = 失败 |
| `data.robot_text` | LLM 生成的完整答案 |
| `data.source` | 项目标识（回显请求中的 `source`） |
| `data.end_flag` | 结束标志，同步接口始终为 `1` |
| `data.status.report_ready_flag` | 报告就绪标志 |
| `data.status.return_task_flag` | 任务返回标志，`1` = 正常 |
| `data.extra_output_params.query` | 原始查询文本 |
| `data.extra_output_params.rewritten_query` | 改写后的查询（多轮对话时有值） |
| `data.extra_output_params.first_frame_time` | 首帧耗时（毫秒），即检索阶段耗时 |
| `data.extra_output_params.final_frame_time` | 总耗时（毫秒） |
| `data.extra_output_params.pages` | 检索到的页面列表，含 chunks |
| `data.extra_output_params.metrics` | 各阶段检索耗时 |

---

## 13. 流式问答（AskBob 协议）

```
POST /rag/chat/stream
```

参数同 [/rag/chat](#12-同步问答（askbob-协议）)，但以 SSE 格式流式返回。每个 SSE 事件的 JSON 结构与同步响应一致，`robot_text` 为**增量文本块**（非累积），`end_flag` 动态更新。

**请求：** 同 `/rag/chat`。

**SSE 事件流：**

```
data: {"code":0,"data":{"robot_text":"根据提供","source":"wiki","end_flag":0,"status":{"report_ready_flag":0,"return_task_flag":1},"extra_output_params":{...}}}

data: {"code":0,"data":{"robot_text":"的资料，**50岁","source":"wiki","end_flag":0,"status":{...},"extra_output_params":{...}}}

data: {"code":0,"data":{"robot_text":"老人**建议","source":"wiki","end_flag":0,"status":{...},"extra_output_params":{...}}}

data: {"code":0,"data":{"robot_text":"根据提供的资料，**50岁老人**建议考虑养老年金保险...","source":"wiki","end_flag":1,"status":{...},"extra_output_params":{...}}}
```

**字段说明：**

| 字段 | 流式行为 |
|------|----------|
| `data.robot_text` | **增量文本块**，客户端自行拼接（最终事件为完整答案） |
| `data.end_flag` | `0`=传输中，`1`=已结束 |
| `data.extra_output_params.first_frame_time` | 传输中为实时耗时，结束事件固定为检索耗时 |
| `data.extra_output_params.final_frame_time` | 实时更新，从请求进入到当前时刻的总耗时 |

**curl 示例：**

```bash
curl -N -X POST http://localhost:8010/rag/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"source":"wiki","user_text":"50岁老人买什么保险合适"}'
```

**JavaScript 前端示例：**

```javascript
const response = await fetch('/rag/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    source: 'wiki',
    user_text: '50岁老人买什么保险合适',
    session_id: 'demo-001',
    trace_id: crypto.randomUUID(),
    user_id: 'user-123'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let fullAnswer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split('\n')) {
    if (!line.startsWith('data: ')) continue;
    const event = JSON.parse(line.slice(6));
    const text = event.data.robot_text;
    fullAnswer += text;
    outputEl.textContent = fullAnswer;

    if (event.data.end_flag === 1) {
      console.log('完成',
        `首字: ${event.data.extra_output_params.first_frame_time}ms`,
        `总耗时: ${event.data.extra_output_params.final_frame_time}ms`);
    }
  }
}
```

---

## 14. 热重载配置

```
POST /admin/reload
```

热重载 YAML 配置，无需重启服务。

**响应：**

```json
{
  "status": "ok",
  "env": "dev",
  "config_path": "/path/to/dev_app_args.yaml",
  "changed_keys": ["log_level", "default_top_pages"]
}
```

---

## 附：检索参数说明

### 三路召回控制

| 参数 | 环境变量 | 说明 |
|------|----------|------|
| `include_es` | `ENABLE_ES_RETRIEVAL` | ES BM25 词汇精确匹配（需配置 `ES_URL`） |
| `include_vector` | `ENABLE_VECTOR_RETRIEVAL` | pgvector 语义相似度（需配置 `PG_HOST`） |
| `include_lexical` | `ENABLE_LOCAL_LEXICAL_RETRIEVAL` | SQLite FTS5 本地全文搜索（始终可用） |

传入 `null` 时使用全局配置（YAML + 环境变量），传入 `true/false` 时覆盖。三路独立运行、RRF 融合。

### 检索流程

```
query
  ├── ES BM25 (include_es)          → chunk_ids + scores
  ├── pgvector (include_vector)     → chunk_ids + scores
  └── SQLite FTS5 (include_lexical) → chunk_ids + scores
              │
              ▼
  RRF 融合 → 候选加载 → 重排序 → 图扩展 → 相邻块扩展 → 上下文构建 → LLM
```

### 配置体系

优先级：**YAML 配置文件 > 环境变量 > 代码默认值**

YAML 按 `ENV` 自动选择：
- `ENV=dev`  → `config/dev_app_args.yaml`
- `ENV=stg`  → `config/stg_app_args.yaml`
- `ENV=prd`  → `config/prd_app_args.yaml`

设置位于 `src/main/python/core/settings.py`，通过 `src/main/python/config/__init__.py` 薄包装导出。
