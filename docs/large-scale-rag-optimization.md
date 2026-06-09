# 大量文档场景 RAG 优化技术方案

## 1. 背景

当前 `llm-rag-v3` 已经按 Wiki-first RAG 思路实现了本地闭环：

```text
wiki/ 或 data/ Markdown 知识层
  -> Markdown-aware chunking
  -> LocalStore JSON 索引
  -> ES BM25 / 本地 lexical / 可选 vector
  -> RRF 融合
  -> 图谱扩展
  -> rerank
  -> 页面级上下文
  -> LLM 回答
```

这个方案适合小中型知识库和本地验证，但当文档量、页面数、chunk 数明显增大后，会遇到性能和召回稳定性问题。

目标是在保持 **Wiki-first RAG** 的前提下，将系统升级为适合大量文档的生产形态：

```text
raw sources -> ingest -> wiki/data Markdown 知识层
查询 -> 检索维护后的 wiki/data 页面
     -> section/page 上下文
     -> LLM 回答
```

## 2. 当前瓶颈

### 2.1 LocalStore 全量 JSON

当前索引存储在：

```text
.rag/indexes/{project_id}.json
```

查询时会一次性加载：

```text
pages, chunks, edges = store.load_index(project_id)
```

问题：

- 大项目启动查询慢。
- 内存占用随全库规模线性增长。
- 索引更新接近全量重写。
- 多项目并发时风险更高。

### 2.2 本地 lexical_search 全量扫描

当前本地词法召回会遍历所有 chunks：

```text
for chunk in chunks:
    score_chunk(...)
```

复杂度近似：

```text
O(全量 chunks × 查询 tokens × 文本长度)
```

数据量大后不能作为主召回路径，只适合小库 fallback。

### 2.3 本地向量召回

早期实现从 JSON 中读取 `chunk.vector` 后逐个 cosine，不适合大库。

当前已加入 SQLite LSH 候选索引：query vector 先按随机投影签名命中候选 chunk，再对候选做 cosine。它避免了默认全库扫描，但仍是本地轻量近似方案；几十万或百万级生产向量召回仍建议迁移到 Qdrant / Milvus / pgvector / Elasticsearch dense_vector。

### 2.4 图谱全量加载

当前 `edges` 全量读入内存后扩展。大图场景下，应只查询 top pages 的一跳邻居。

### 2.5 页面级上下文在长页面上成本高

页面级上下文能提升事实完整性，但大量文档时长页面多、召回候选多，不能无脑塞整页，否则：

- prompt 成本高；
- 噪声变多；
- 关键 section 被截断；
- LLM 回答变啰嗦。

## 3. 目标架构

大量文档场景推荐目标形态：

```text
                 raw docs
                    |
                 ingest
                    |
          wiki/data Markdown pages
                    |
        incremental index worker
        /        |          \
      DB       ES/FTS      Vector DB
 pages/chunks  lexical      ANN
 graph/manifest
        \        |          /
          retrieval pipeline
             RRF + graph
                rerank
        section/page context
                LLM
```

核心原则：

- `chunk` 用于召回和排序。
- `page/section` 用于最终上下文。
- 大量文档时不加载全库，只拉候选。
- ES/FTS 和向量库承担主召回。
- 本地全量 lexical scan 只保留为小库模式。

## 4. 存储层优化

### 4.1 从 JSON 切到数据库

将 LocalStore 拆成表：

```text
pages
chunks
graph_edges
manifest_files
```

建议 schema：

```text
pages(
  project_id,
  page_id,
  path,
  title,
  type,
  sources,
  wikilinks,
  content,
  content_sha256,
  mtime,
  indexed_at
)

chunks(
  project_id,
  chunk_id,
  page_id,
  path,
  title,
  heading_path,
  type,
  sources,
  content,
  chunk_index,
  prev_chunk_id,
  next_chunk_id,
  token_estimate
)

graph_edges(
  project_id,
  source_page_id,
  target_page_id,
  edge_type,
  weight
)

manifest_files(
  project_id,
  path,
  sha256,
  mtime,
  chunk_count,
  indexed_at
)
```

可选实现：

- 本地/轻量：SQLite。
- 服务端生产：Postgres。
- 高并发：Postgres + 连接池 + 后台 worker。

### 4.2 查询只拉候选

查询流程应变为：

```text
retrieval sources -> top chunk/page ids
DB -> load candidate pages/chunks/edges
context builder -> pack selected context
```

避免：

```text
load all pages
load all chunks
load all edges
```

## 5. 词法召回优化

### 5.1 ES/OpenSearch 作为主词法召回

大量文档时，ES BM25 应从“可选”升级为生产主召回之一。

当前 ES 召回率低时，优先优化 analyzer 和 query，不要退回全量扫描。

建议字段：

```text
title.keyword
title.text
title.bigram
heading_path.text
heading_path.bigram
path.text
sources.keyword
content.text
content.bigram
```

建议 query 结构：

```text
should:
  - term title.keyword exact boost
  - match_phrase title boost
  - match_phrase heading_path boost
  - multi_match best_fields
  - multi_match most_fields on bigram fields
minimum_should_match: 1
filter:
  - project_id
```

不要用过强的 `must` 限制中文查询，否则分词不匹配时召回会变低。

### 5.2 本地 lexical 改为倒排索引

如果需要本地 fallback，不应全量遍历 chunks，而应预建倒排索引：

```text
token -> chunk_ids/page_ids
bigram -> chunk_ids/page_ids
trigram -> chunk_ids/page_ids
title_phrase -> page_ids
```

查询时：

```text
query tokens -> postings union/intersection
candidate ids -> scoring
top candidates -> RRF
```

复杂度从：

```text
O(全量 chunks)
```

降低到：

```text
O(命中 postings + 候选重排)
```

### 5.3 推荐本地 FTS 方案

可选：

- SQLite FTS5：轻量、易部署，适合本地服务。
- Tantivy：性能强，适合较大本地索引。
- Whoosh：纯 Python，但大数据性能一般。

推荐当前项目优先选：

```text
SQLite FTS5 + 自定义中文 bigram/trigram 字段
```

原因：

- 和 Python 服务集成简单。
- 不需要额外服务。
- 比全量扫描提升明显。

### 5.4 中文 token 策略

保险健康服务类查询常见短语：

```text
家庭医生服务
重疾专案管理
门诊预约协助
等待期
适用对象
```

建议索引：

```text
原词
中文 bigram
中文 trigram
单字 fallback
同义别名
```

字段权重建议：

```text
title_exact       300
title_phrase      120
path_phrase        80
heading_phrase     60
content_phrase     30
title_bigram       12
heading_bigram      8
content_bigram      2
```

## 6. 向量召回优化

当前本地 vector search 是全量 cosine，不适合大库。

生产建议：

```text
Qdrant / Milvus / pgvector / Elasticsearch dense_vector
```

向量 payload：

```text
project_id
page_id
chunk_id
path
title
heading_path
type
sources
```

查询流程：

```text
query -> embedding
ANN topK chunks
group by page_id
page-level score aggregation
```

建议参数：

```text
vector_top_k = 300~1000
per_page_anchor_chunks = 1~3
```

## 7. Hybrid 融合

继续使用 RRF，避免不同召回源分数尺度不一致。

```text
rrf_score =
  1 / (60 + lexical_rank)
  + 1 / (60 + vector_rank)
```

大量文档时可以融合多个 source：

```text
RRF({
  es_bm25,
  title_exact,
  local_fts,
  vector
})
```

候选控制：

```text
每路召回 top 300~1000 chunks/pages
RRF 后保留 top 300 candidates
page 聚合后保留 top 100 pages
rerank 前保留 top 100~200 chunks/pages
```

## 8. 图谱扩展优化

当前图谱扩展全量加载 edges。大数据时应改成按候选查询：

```sql
select *
from graph_edges
where project_id = ?
  and source_page_id in (...)
  and weight >= 2.0
order by weight desc
limit ...
```

扩展策略保持：

```text
P0 标题命中
P1 内容命中
P2 图谱扩展
P3 overview fallback
```

图谱扩展只对 top pages 做一跳扩展，避免大图遍历。

## 9. 上下文打包优化

大量文档时，不建议所有命中页面都整页打包。

推荐 section-aware context：

```text
短页面：整页
长页面：命中 section 优先
       + prev/next section
       + 页面摘要
       + 必要规则段/表格
```

预算保持：

```text
index_budget = max_context_size * 0.05
page_budget = max_context_size * 0.50
response_reserve = max_context_size * 0.15
```

单页策略：

```text
page <= 8000 chars:
  pack full page

page > 8000 chars:
  pack matched section
  pack neighboring sections
  pack page summary if available
  truncate by per_page_cap
```

这样能降低 prompt 成本，也能避免长页面前半部分挤掉真正命中的规则段。

## 10. 增量索引

当前已用 manifest 复用 unchanged 文件的旧 chunks/vectors，避免重复 chunking 和 embedding；SQLite、ES 已支持 pages/chunks 路径级局部 upsert/delete；JSON 仍保留完整快照。下一步生产模式应继续把 graph edges 和 manifest 迁移到真正 DB 表。

生产模式应实现：

```text
新增文件:
  insert page/chunks/vectors/edges

修改文件:
  delete old chunks/vectors/edges
  update page
  insert new chunks/vectors/edges

删除文件:
  delete page/chunks/vectors/edges

chunker_version 变化:
  rebuild chunks/vectors

embedding_model 变化:
  rebuild vectors
```

要求：

- `page_id` 稳定。
- `chunk_id` 可追踪。
- 索引任务可恢复。
- embedding/rerank 失败可重试。
- ES/向量库/DB 更新尽量事务化或可补偿。

## 11. Rerank 优化

大量文档时 rerank 不能直接处理大量候选。

推荐：

```text
lexical/vector 各召回 top 500~1000
RRF 融合
page 聚合 top 100
每页取 top 1~3 anchor chunks
rerank top 100~200
最终选 top 8~12 pages/sections
```

避免：

```text
对几千上万个候选直接 rerank
```

## 12. 运行模式

建议拆成两个模式。

### 12.1 Local Prototype Mode

适合：

```text
几十到几千个 Markdown 页面
几千到几万个 chunks
本地验证
```

能力：

```text
LocalStore JSON
本地 lexical fallback
可选 ES
可选本地 vector
页面级 context
```

### 12.2 Production Scale Mode

适合：

```text
几十万 chunks 以上
多项目并发
频繁增量更新
低延迟线上服务
```

能力：

```text
DB pages/chunks/edges/manifest
ES/OpenSearch lexical
SQLite FTS5/Tantivy fallback lexical
Vector DB ANN
按候选加载 graph
section-aware context
异步增量 indexing
```

## 13. 分阶段落地计划

### M1：保留当前本地模式

- 当前 LocalStore 继续可用。
- 本地 lexical scan 只用于小库 fallback。
- 增加配置区分 local / production。

### M2：DB 化

- pages/chunks/edges/manifest 分表。
- 查询只加载候选。
- 索引任务写 DB。

### M3：词法召回升级

- ES analyzer/query 优化。
- 增加 SQLite FTS5 或 Tantivy fallback。
- 禁用大库全量 lexical scan。

### M4：向量库升级

- 接入 Qdrant / Milvus / pgvector。
- payload 带 project/page/chunk metadata。
- ANN 查询后 page 聚合。

### M5：上下文打包升级

- 短页整页。
- 长页 section-first。
- overview fallback 保留。
- 支持页面摘要。

### M6：评测与监控

指标：

```text
Recall@K
MRR
nDCG@K
fallback_rate
context_truncation_rate
answer citation rate
latency p50/p95/p99
```

## 14. 推荐优先级

如果马上要支持大量文档，优先做三件事：

```text
1. 禁止本地 lexical 全量扫描作为大库主路径。
2. LocalStore JSON 改为 DB/索引分层存储。
3. 长页面上下文改成 section-first，而不是总是整页。
```

当前已落地：大库本地 lexical 全量扫描阈值保护、SQLite FTS5 fallback、SQLite 候选级 pages/chunks/graph 读取、长页面 section-first 打包、unchanged 文件 chunks/vectors 复用、SQLite/ES pages/chunks 路径级局部 upsert/delete。

最终目标：

```text
Wiki-first
+ ES/FTS lexical
+ Vector DB
+ DB graph
+ RRF
+ rerank
+ section-aware context
```
