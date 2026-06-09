# LLM RAG V3 生产级 RAG 设计方案

## 1. 项目定位

`llm-wiki-rag` 是面向保险健康服务知识库的本地优先 RAG 服务。当前代码已经实现从维护后的 Markdown 知识层到索引、混合检索、图谱扩展、重排、页面级上下文打包和最终问答的完整闭环。

核心链路：

```text
wiki/**/*.md 或 data/**/*.md
  -> Markdown/frontmatter 解析
  -> Markdown-aware chunking
  -> 本地 JSON 索引 + 可选 Elasticsearch
  -> BM25/词法召回 + 向量召回
  -> RRF 融合
  -> 页面聚合
  -> 图谱扩展
  -> DashScope qwen3-rerank 精排
  -> 页面级上下文打包
  -> OpenAI SDK 调 DashScope compatible chat/completions 问答
```

设计目标：

- 让回答质量靠近 `LLM-wiki-black` 的页面级上下文策略，避免只命中局部 chunk 导致规则、限制、次数、适用对象丢失。
- 保留 v3 已有的混合召回、RRF、图谱扩展和 rerank 能力。
- ES BM25 与向量召回均为可选能力，可通过环境变量或单次请求关闭。
- 在 Elasticsearch、DashScope 或 reranker 不可用时保持可降级。
- 为后续异步任务队列、评测集、监控和生产部署预留清晰边界。

## 2. 当前实现架构

```text
┌──────────────────────┐
│ Markdown Knowledge   │
│ wiki/ or data/ *.md  │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│ Index Worker          │
│ scanner/parser/chunk  │
└───────┬──────────────┘
        │
        ├────────────────────────────────┐
        v                                v
┌──────────────────────┐        ┌──────────────────────┐
│ LocalStore            │        │ Elasticsearch optional│
│ .rag/indexes/*.json   │        │ pages/chunks/edges    │
│ .rag/manifests/*.json │        └──────────┬───────────┘
└──────────┬───────────┘                   │
           │                               │
           └──────────────┬────────────────┘
                          v
┌─────────────────────────────────────────────────────┐
│ Retrieval Pipeline                                  │
│ BM25/lexical + vector + RRF + graph + rerank        │
└──────────┬──────────────────────────────────────────┘
           v
┌─────────────────────────────────────────────────────┐
│ Page-level Context Builder                          │
│ selected pages -> numbered page bodies              │
└──────────┬──────────────────────────────────────────┘
           v
┌──────────────────────┐
│ Answer API / CLI      │
│ OpenAI SDK + DashScope│
└──────────────────────┘
```

当前主要入口：

- CLI：`python3 -m app.cli index|process-data|query|answer`
- API：`GET /health`
- API：`POST /data/process`
- API：`POST /rag/index/project`
- API：`GET /jobs/{job_id}`
- API：`POST /rag/context`
- API：`POST /rag/search/debug`
- API：`POST /rag/answer`

## 3. 快速使用

### 3.1 配置环境

```bash
cd /Users/mickey/project/PA-ALG/llm-wiki-rag

export DASHSCOPE_API_KEY=sk-xxx
export LLM_MODEL=vanchin/deepseek-v4-pro
export LLM_REASONING_EFFORT=high
export DASHSCOPE_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 推荐先关闭向量召回，跑通 Wiki-first 页面级 RAG。
export ENABLE_VECTOR_RETRIEVAL=false
```

检查当前进程实际读取的非密钥配置：

```bash
python3 -m app.cli config
```

### 3.2 索引 data 知识库

```bash
python3 -m app.cli index --project-id pingan-zhenxiang --path data --force --no-embeddings
```

如果要启用向量召回，去掉 `--no-embeddings` 并确保 `DASHSCOPE_API_KEY` 可用于 embedding。

### 3.3 查询和问答

```bash
python3 -m app.cli query \
  --project-id pingan-zhenxiang \
  --query "家庭医生服务次数和适用对象是什么" \
  --no-vector

python3 -m app.cli answer \
  --project-id pingan-zhenxiang \
  --query "家庭医生服务次数和适用对象是什么" \
  --no-vector
```

### 3.4 API 服务

```bash
python3 -m pip install -e ".[api]"
python3 -m app.main --host 0.0.0.0 --port 8010 --reload
```

```bash
curl -X POST http://127.0.0.1:8010/rag/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "pingan-zhenxiang",
    "query": "家庭医生服务次数和适用对象是什么",
    "include_vector": false
  }'
```

## 4. 数据模型

### 4.1 Page

```json
{
  "project_id": "pingan-zhenxiang",
  "page_id": "entities-family-doctor",
  "path": "entities/家庭医生服务.md",
  "title": "家庭医生服务",
  "type": "entity",
  "sources": ["平安臻享家医服务手册.md"],
  "wikilinks": ["家庭医生服务组"],
  "content": "去除 frontmatter 后的完整页面正文",
  "metadata": {},
  "content_sha256": "abc...",
  "mtime": 1710000000,
  "chunk_count": 4,
  "indexed_at": "2026-06-02T10:00:00+00:00"
}
```

### 4.2 Chunk

```json
{
  "project_id": "pingan-zhenxiang",
  "page_id": "entities-family-doctor",
  "chunk_id": "entities-family-doctor#0000",
  "path": "entities/家庭医生服务.md",
  "title": "家庭医生服务",
  "heading_path": "家庭医生服务 > 服务次数",
  "type": "entity",
  "sources": ["平安臻享家医服务手册.md"],
  "content": "chunk 正文",
  "chunk_index": 0,
  "prev_chunk_id": null,
  "next_chunk_id": "entities-family-doctor#0001",
  "token_estimate": 300,
  "vector": []
}
```

### 4.3 GraphEdge

```json
{
  "project_id": "pingan-zhenxiang",
  "source_page_id": "entities-family-doctor",
  "target_page_id": "concepts-family-doctor-group",
  "edge_type": "wikilink",
  "weight": 3.0
}
```

### 4.4 Manifest

```json
{
  "project_id": "pingan-zhenxiang",
  "chunker_version": "markdown-aware-local-v1",
  "embedding_model": "text-embedding-v4",
  "files": {
    "entities/家庭医生服务.md": {
      "sha256": "...",
      "mtime": 1710000000,
      "chunk_count": 4,
      "indexed_at": "2026-06-02T10:00:00+00:00",
      "status": "indexed"
    }
  }
}
```

## 5. 索引流程

当前索引由 `app.indexing.worker.index_project` 同步执行。

```text
1. 解析知识层根目录：
   - project_path/wiki 存在时扫描 wiki/
   - 否则 project_path/data 存在时扫描 data/
   - 否则扫描传入的 project_path
2. 读取文件正文
3. 计算 sha256 / mtime
4. 如果 manifest 判定文件 unchanged 且旧索引可用，复用旧 Page / Chunk / vector
5. 对新增或变更文件解析 frontmatter、title、type、sources、wikilinks
6. 构造 Page
7. Markdown-aware chunking，生成 Chunk 和 prev/next 关系
8. 使用 DashScope text-embedding-v4 为 chunk 生成向量
   - DashScope 不可用时降级为 hash embedding
9. 根据 wikilink、shared_source、same_type 生成 GraphEdge
10. 写入 LocalStore:
   - .rag/indexes/{project_id}.json
   - .rag/indexes/{project_id}.sqlite3
   - .rag/manifests/{project_id}.json
11. 如果 ES_URL 可用，同步写入 Elasticsearch:
   - llm_wiki_pages
   - llm_wiki_chunks
   - llm_wiki_graph_edges
```

当前 manifest 已用于复用 unchanged 文件的旧 chunks/vectors，减少重复 chunking 和 embedding。JSON 仍写完整快照；SQLite 和 ES 已支持按变更/删除 path 做 pages/chunks 局部 delete/upsert。Graph edges 因依赖全局页面关系，当前仍按项目级边集合刷新。

## 6. Markdown-aware Chunking

当前参数：

```text
target_chars = 1200
max_chars = 1500
min_chars = 300
overlap_chars = 50
```

切分策略：

```text
1. 按 Markdown heading 拆 section
2. 按空行拆 block
3. 超长 block:
   - 表格按行打包
   - 中文/英文句子按句末符号打包
   - 无法识别时硬切
4. 相邻短 block 合并到 target_chars
5. 每个 chunk 保留 heading_path
6. 建立 prev_chunk_id / next_chunk_id
7. 相邻 chunk 按 overlap_chars 添加前文 overlap
```

注意：chunk 仍用于召回、RRF、rerank 和命中解释；最终上下文打包不再只拼 chunk，而是优先打包选中页面正文。

## 7. Elasticsearch 设计

Elasticsearch 是可选能力。不可用时系统会使用本地词法召回和本地向量数据继续工作。

### 7.1 pages index

```text
project_id keyword
page_id keyword
path keyword
title text + keyword
type keyword
sources keyword
wikilinks keyword
content text
content_sha256 keyword
mtime double
indexed_at date
```

### 7.2 chunks index

```text
project_id keyword
page_id keyword
chunk_id keyword
path keyword
title text + keyword
heading_path text + keyword
type keyword
sources keyword
content text
chunk_index integer
prev_chunk_id keyword
next_chunk_id keyword
token_estimate integer
```

BM25 查询字段权重：

```text
title^5
heading_path^3
path^2
sources^2
content
```

中文生产环境建议补充 IK 或同等级中文 analyzer。当前代码未强制配置 analyzer。

## 8. 检索流程

`retrieve_context` 默认参数：

```text
max_context_size = 204800
top_pages = 8
bm25_top_k = 100
vector_top_k = 100
rerank_top_k = 50
include_es = None
include_vector = None
include_graph = true
include_neighbor_chunks = true
```

`include_es` 和 `include_vector` 为请求级覆盖项：

```text
None  -> 使用环境变量默认值
true  -> 本次请求启用对应召回
false -> 本次请求关闭对应召回
```

完整链路：

```text
1. 从 LocalStore 加载 pages/chunks/edges
2. 如果 ES 召回启用，执行 Elasticsearch BM25 chunk search
3. 如果 ES 关闭或不可用，执行本地 lexical fallback
4. 如果向量召回启用，执行本地 vector search
   - query 使用 DashScope embedding
   - DashScope 不可用时 hash embedding 降级
   - 向量召回关闭时不会生成 query embedding
5. BM25/lexical/vector 结果做 RRF 融合
6. chunk 结果聚合成 page 排名
7. 基于图谱扩展相关页面
8. 取融合候选 chunk + 图谱扩展页面最佳 chunk
9. 使用 DashScope qwen3-rerank 精排
   - rerank 不可用时使用本地 token overlap 打分
10. 计算最终 page score
11. 对选中页面做邻接 chunk 扩展，用于命中解释
12. 如果无命中，尝试加入 overview.md 兜底
13. 使用页面级上下文打包返回 numbered context
```

降级标记会写入响应的 `fallback_reasons`。

## 9. 本地词法召回

当 Elasticsearch 不可用时，`lexical_search` 会在本地 chunks 上执行中文友好的轻量召回。

策略：

- 查询短语精确命中 title / heading / content 加权。
- 中文长词拆 bigram 和单字，提高短中文查询召回。
- `entity`、`concept` 页面有轻微 boost。
- `source` 页面轻微降权，避免原始资料页淹没结构化实体页。

## 10. 向量召回

索引时 chunk embedding 输入：

```text
Title: {title}
Section: {heading_path}
Content: {content}
```

查询时：

```text
1. query -> embedding
2. 默认使用 SQLite vector_lsh 表按随机投影签名召回候选 chunk
3. 只对候选 chunk.vector 做 cosine similarity
4. 返回 top_k chunk_id / score
```

当前向量仍保存在 JSON chunk 对象和 SQLite chunks.vector 中，同时写入 SQLite LSH 候选索引。SQLite LSH 是零依赖本地 ANN 近似方案；生产演进仍建议迁移到 Qdrant、Milvus、pgvector 或 Elasticsearch dense_vector。

## 11. RRF 融合

BM25、lexical fallback 和 vector 的 chunk 排名通过 RRF 融合。

```text
rrf_score = Σ 1 / (k + rank)
```

设计原因：

- 不直接混加不同检索器的原始分数。
- 双路命中的 chunk 自然得分更高。
- 单路强命中仍可进入候选。

## 12. Page 聚合

RRF 后先从 chunk 聚合到 page：

```text
page_score =
  max(chunk_rrf_score)
  + 0.08 * sum(next_best_4_chunk_scores)
  + title_boost
```

类型权重：

```text
entity  * 1.18
concept * 1.08
source  * 0.72
```

每个 page 保留：

- `anchor_chunks`: top 1-3 命中 chunk
- `matched_sources`
- `score`
- `title`
- `path`

## 13. 图谱扩展

图谱边来源：

```text
wikilink      weight = 3.0
shared_source weight = 4.0
same_type     weight = 1.0
```

扩展规则：

```text
输入: page aggregation 后 top 20 pages
每个 page 最多扩展 3 个邻居
edge.weight >= 2.0 才加入
已命中页面不重复加入
```

图谱扩展页面会选择该页面内与 query 最相关的 chunk 作为 rerank 候选，但最终上下文仍按页面正文打包。

如果检索与图谱扩展均无候选，pipeline 会选择 `overview.md` 或标题为 `Overview` 的页面作为 P3 fallback。

## 14. Reranker 与最终排序

rerank 输入：

```text
Title: {chunk.title}
Section: {chunk.heading_path}
Content: {chunk.content}
```

当前最终分数：

```text
final_score =
  0.25 * normalized_page_rrf
  + 0.35 * normalized_chunk_rrf
  + 0.20 * rerank_score
  + 0.10 * title_match_boost
  + 0.10 * graph_score
  + 0.05 * metadata_boost
```

`title_match_boost` 用于让标题精确命中或标题短语命中的页面更靠前，策略上贴近 `LLM-wiki-black` 的“标题命中优先”。

reranker 不可用时，系统使用本地 query/content/title/heading token overlap 打分。

## 15. 页面级上下文打包

这是当前版本相对早期 v3 的关键调整。

早期策略：

```text
命中 chunk -> 拼接少量 chunk / 邻接 chunk
```

当前策略：

```text
选中 page -> 按预算打包该 page 正文
```

预算：

```text
DEFAULT_MAX_CTX = 204800
index_budget = max_context_size * 0.05
page_budget = max_context_size * 0.50
response_reserve = max_context_size * 0.15
per_page_cap = min(page_budget, max(5000, page_budget * 0.30))
```

返回结构：

```json
{
  "pages": [
    {
      "number": 1,
      "page_id": "entities-family-doctor",
      "title": "家庭医生服务",
      "path": "entities/家庭医生服务.md",
      "score": 0.92,
      "source": "hybrid",
      "chunks": [
        {
          "chunk_id": "entities-family-doctor#0000",
          "heading_path": "家庭医生服务 > 服务次数",
          "content": "命中 chunk 正文",
          "score": 0.92
        }
      ]
    }
  ],
  "page_list": "[1] 家庭医生服务 (entities/家庭医生服务.md)",
  "pages_context": "### [1] 家庭医生服务\nPath: ...\n\n完整页面正文...",
  "purpose": "Use the numbered context to answer the user query with source-aware grounding.",
  "index": "[1] 家庭医生服务 (...)",
  "source": "page_context_es_bm25_text_embedding_v4_rrf_graph_qwen3_rerank"
}
```

设计取舍：

- 优点：模型能看到完整规则上下文，问答效果更接近 wiki-black。
- 优点：减少“命中了服务名但漏掉限制/等待期/适用对象”的问题。
- 代价：单次 prompt 更长，对模型上下文窗口和调用成本要求更高。
- 保留 chunk metadata，用于 debug、来源解释和后续 UI 高亮。

## 16. Answer 生成

`/rag/answer` 在 `/rag/context` 之上调用 OpenAI SDK 的 DashScope compatible `chat/completions`。

系统提示词原则：

- 只依据 numbered context 回答。
- 上下文不足时说明“不足以确认”。
- 事实、规则、次数、适用对象、流程、限制必须引用来源编号。
- 使用中文回答。
- 末尾追加隐藏引用注释：`<!-- cited: 1, 3 -->`

当前模型默认：

```text
LLM_MODEL=vanchin/deepseek-v4-pro
LLM_REASONING_EFFORT=
```

生产或本地调试推理模型时建议显式设置：

```text
LLM_REASONING_EFFORT=high
```

## 17. API 设计

### 17.1 健康检查

```http
GET /health
```

### 17.2 索引项目

```http
POST /rag/index/project
```

请求：

```json
{
  "project_id": "pingan-zhenxiang",
  "project_path": "data",
  "force": false,
  "build_embeddings": true
}
```

响应为同步任务结果，包含 `job_id` 和索引统计。

### 17.3 处理默认数据

```http
POST /data/process
```

请求：

```json
{
  "project_id": "pingan-zhenxiang",
  "data_path": "/Users/mickey/project/PA-ALG/llm-wiki-rag/data",
  "force": true,
  "build_embeddings": true
}
```

### 17.4 查询任务状态

```http
GET /jobs/{job_id}
```

当前任务队列是进程内 `JOBS` 字典，适合本地开发和单进程服务。生产环境需要替换成 Redis/Celery/Dramatiq 或数据库任务表。

### 17.5 构建 RAG 上下文

```http
POST /rag/context
```

请求：

```json
{
  "project_id": "pingan-zhenxiang",
  "query": "家庭医生服务次数和适用对象是什么",
  "max_context_size": 204800,
  "top_pages": 8,
  "bm25_top_k": 100,
  "vector_top_k": 100,
  "rerank_top_k": 50,
  "include_graph": true,
  "include_neighbor_chunks": true
}
```

### 17.6 检索调试

```http
POST /rag/search/debug
```

在普通 context 响应基础上附加：

```text
bm25
lexical
vector
rrf
pages
graph_expansions
selected
```

### 17.7 直接问答

```http
POST /rag/answer
```

请求体与 `/rag/context` 相同，响应包含：

```json
{
  "answer": "...",
  "llm_error": null,
  "context": {}
}
```

## 18. 配置

当前代码支持：

```text
RAG_HOST=0.0.0.0
RAG_PORT=8010
WIKI_DATA_PATH=/Users/mickey/project/PA-ALG/llm-wiki-rag/data
RAG_STORAGE_DIR=.rag

ES_URL=http://localhost:9200
ES_USER=
ES_PASSWORD=
ES_INDEX_PREFIX=llm_wiki
ENABLE_ES_INDEXING=false
ENABLE_ES_RETRIEVAL=true
ENABLE_VECTOR_RETRIEVAL=true
ENABLE_LOCAL_LEXICAL_RETRIEVAL=true
LOCAL_LEXICAL_MAX_CHUNKS=50000
LOCAL_VECTOR_MAX_CHUNKS=50000
VECTOR_SEARCH_BACKEND=sqlite_lsh
VECTOR_LSH_TABLES=8
VECTOR_LSH_BITS=12
VECTOR_LSH_CANDIDATE_MULTIPLIER=20
BUILD_EMBEDDINGS=false
DEFAULT_MAX_CONTEXT_SIZE=204800
DEFAULT_TOP_PAGES=8
DEFAULT_BM25_TOP_K=300
DEFAULT_VECTOR_TOP_K=100
DEFAULT_RERANK_TOP_K=100
RAG_LOG_LEVEL=INFO
RAG_LOG_FORMAT=json
RAG_LOG_FILE=

DASHSCOPE_API_KEY=sk-xxx
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=0
RERANK_MODEL=qwen3-rerank
LLM_MODEL=vanchin/deepseek-v4-pro
LLM_REASONING_EFFORT=high
DASHSCOPE_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_RERANK_BASE_URL=https://dashscope.aliyuncs.com/compatible-api/v1
DASHSCOPE_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Chat 调用使用 OpenAI SDK 的 DashScope compatible 模式，并读取 `DASHSCOPE_API_KEY`。代码不应包含默认 DashScope key，生产环境必须从环境变量或 Secret 注入。

## 19. 错误处理与降级

当前降级：

```text
Elasticsearch 不可用 -> SQLite FTS5 fallback -> 小库 lexical fallback
ES 召回关闭 -> SQLite FTS5 fallback -> 小库 lexical fallback
向量召回关闭 -> 跳过 vector search 和 query embedding
chunk 数超过 LOCAL_LEXICAL_MAX_CHUNKS -> 跳过全量 lexical scan
chunk 数超过 LOCAL_VECTOR_MAX_CHUNKS -> 跳过本地全量 vector cosine
DashScope embedding 不可用 -> hash embedding
DashScope rerank 不可用 -> 本地 token overlap rerank
DashScope chat 不可用 -> /rag/answer 返回 llm_error
项目索引不存在 -> FileNotFoundError，经 API 转 404
```

响应中的 `fallback_reasons` 用于解释实际走了哪些降级路径。

生产建议：

- 对 ES/DashScope 错误做结构化日志。
- 区分临时超时、配置缺失、认证失败。
- 对 embedding/rerank/chat 增加重试、超时和熔断。
- 对 API 返回统一错误 schema。

## 20. 测试策略

当前已有单元测试覆盖：

- frontmatter 解析
- chunker heading_path / neighbor
- retrieval 基础链路
- context builder 页面级正文打包
- DashScope fallback
- Elasticsearch BM25 基础行为

建议继续补充：

- 标题命中优先排序测试。
- 图谱扩展排序测试。
- ES 不可用时 lexical fallback 的中文召回测试。
- 长页面预算截断测试。
- `/rag/search/debug` 快照测试。
- `/rag/answer` mock LLM 引用编号测试。

## 21. 评测设计

建议新增 `eval/` 模块和 JSONL 评测集：

```json
{
  "query": "家庭医生服务次数和适用对象是什么",
  "expected_pages": ["entities/家庭医生服务.md"],
  "expected_terms": ["不限次", "适用对象"],
  "must_cite": true
}
```

检索指标：

```text
Recall@K
MRR
nDCG@K
expected_page_hit_rate
fallback_rate
context_truncation_rate
```

回答指标：

```text
引用编号存在率
关键事实覆盖率
幻觉率
不足以确认误判率
```

## 22. 生产化演进

### M1 已完成：本地 RAG 闭环

- Markdown parser
- chunker
- LocalStore
- manifest
- ES 可选索引
- BM25/lexical/vector/RRF
- graph expansion
- rerank
- page-level context
- answer API

### M2 检索质量增强

- 标题命中和页面命中策略参数化。
- source/entity/concept 类型权重可配置。
- 更强中文 analyzer。
- query rewrite / query expansion。
- 页面正文打包支持从命中 section 开始的“相关区间优先”，超长页再补全文。

### M3 任务队列和增量索引

- 进程内 `JOBS` 替换为 Redis/Celery/Dramatiq。
- manifest 真正驱动增量新增、修改、删除。
- ES 删除旧 page/chunk/edge。
- embedding 批处理和失败重试。

### M4 外部向量库

- Qdrant / Milvus / Elasticsearch dense_vector。
- 向量 payload 使用 project_id/page_id/chunk_id 过滤。
- 支持多 embedding model 版本并行回滚。

### M5 可观测性与评测

- structured logging。
- Prometheus metrics。
- debug trace 持久化。
- eval runner。
- CI 回归评测。

### M6 安全与部署

- Secret 管理。
- project_path 访问范围限制。
- API 鉴权。
- Docker Compose / Kubernetes。
- 多进程 worker。

## 23. 关键设计结论

当前 v3 的推荐策略是：

```text
chunk 用于召回和排序，page 用于最终上下文。
```

这样比“只拼命中 chunk”更适合保险健康服务类知识库，因为用户常问的是权益、次数、限制、适用对象、等待期和流程组合问题；这些答案往往分布在同一页面的多个段落里。页面级上下文会增加 prompt 成本，但能显著降低“召回到了服务名，却漏掉关键规则”的概率。
