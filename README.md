# LLM RAG V3

基于 `docs/production-rag-design.md` 实现的本地优先 RAG 系统，适配 Markdown 知识库格式。

## 框架结构

```
src/main/python/
├── api/             # HTTP 路由（routes.py）
├── config/          # 配置薄包装 → core/settings.py
├── core/            # 核心配置（Settings 类）
├── db/              # 数据层（ES, pgvector, SQLite, DashScope）
├── models/          # 数据模型
├── services/        # 服务层（RAGService）
├── steps/           # 处理管道
│   ├── agents/       #   AgnoRAGAgent（问答编排）
│   ├── indexing/     #   索引管道（chunker, embedding, graph, worker）
│   └── retrieval/    #   检索管道（pipeline, retrievers, reranker, context）
├── tools/           # CLI 工具
└── utils/           # 工具函数
```

## 核心能力

- Markdown 知识库扫描、frontmatter 解析、wikilink 提取
- Markdown-aware chunking（保留标题路径、前后 chunk 链）
- 增量索引（manifest 对比 sha256，复用未变化文件）
- 三路混合检索：ES BM25 + pgvector 语义 + SQLite FTS5 词召回
- RRF 融合 → DashScope qwen3-rerank 精排 → 图扩展 → 相邻块扩展 → 上下文打包
- Agno Agent 多轮对话 + DashScope LLM 答案生成

## 快速使用

```bash
# 索引
python -m src.main.python.tools.cli index --project-id my-project --path data --force

# 检索
python -m src.main.python.tools.cli query --project-id my-project --query "家庭医生服务有哪些内容？"

# 问答
python -m src.main.python.tools.cli answer --project-id my-project --query "家庭医生服务有哪些内容？"
```

多轮对话复用 `session_id`：

```bash
python -m src.main.python.tools.cli answer --project-id my-project --session-id demo-001 --query "家庭医生服务次数是多少"
python -m src.main.python.tools.cli answer --project-id my-project --session-id demo-001 --query "这个服务适用对象是谁"
```

## 配置

配置优先级：**环境变量 > YAML 配置文件 > 代码默认值**。YAML 按 `ENV` 环境变量自动选择：

```bash
ENV=dev  → config/dev_app_args.yaml
ENV=stg  → config/stg_app_args.yaml
ENV=prd  → config/prd_app_args.yaml
```

### 常用环境变量

```bash
# DashScope API
export DASHSCOPE_API_KEY=sk-xxx
export EMBEDDING_MODEL=text-embedding-v4
export RERANK_MODEL=qwen3-rerank
export LLM_MODEL=qwen-plus

# 功能开关
export ENABLE_ES_RETRIEVAL=false
export ENABLE_VECTOR_RETRIEVAL=false
export ENABLE_LOCAL_LEXICAL_RETRIEVAL=true

# ES 配置
export ES_URL=http://localhost:9200
export ES_INDEX_PREFIX=llm_wiki

# pgvector 配置
export PG_HOST=localhost
export PG_PORT=5432
export PG_DATABASE=rag

# 检索参数
export DEFAULT_MAX_CONTEXT_SIZE=204800
export DEFAULT_TOP_PAGES=8
export DEFAULT_BM25_TOP_K=300
export DEFAULT_VECTOR_TOP_K=100
export DEFAULT_RERANK_TOP_K=100

# Agent
export ENABLE_AGNO_AGENT=true
export AGNO_SESSION_DB_PATH=.rag/agents/sessions.db
export AGNO_HISTORY_RUNS=6

# 日志
export RAG_LOG_LEVEL=INFO
export RAG_LOG_FILE=logs/rag.log
```

## API 服务

```bash
# 安装依赖
pip install -e ".[api]"

# 启动
python -m src.main.python.main --host 0.0.0.0 --port 8010

# 或直接用 uvicorn
uvicorn src.main.python.main:app --reload --port 8010
```

### 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/health/ready` | 深度就绪检查 |
| POST | `/rag/index/project` | 索引项目 |
| POST | `/data/process` | 数据处理快捷方式 |
| GET | `/jobs/{job_id}` | 查询作业状态 |
| POST | `/rag/context` | 检索上下文 |
| POST | `/rag/search/debug` | 调试检索 |
| POST | `/rag/answer` | 同步问答 |
| POST | `/rag/answer/stream` | 流式问答（SSE） |
| POST | `/rag/chat` | AskBob 协议同步问答 |
| POST | `/rag/chat/stream` | AskBob 协议流式问答（SSE） |
| GET | `/sessions/{session_id}` | 获取会话 |
| DELETE | `/sessions/{session_id}` | 清除会话 |
| POST | `/admin/reload` | 热重载配置 |

详细 API 文档见 [docs/API.md](docs/API.md)。

### 请求示例

```json
{
  "project_id": "my-project",
  "query": "家庭医生服务次数",
  "include_es": false,
  "include_vector": false,
  "top_pages": 5
}
```

## 索引持久化

- SQLite: `.rag/indexes/{project_id}.sqlite3`
- JSON 快照: `.rag/indexes/{project_id}.json`
- Manifest: `.rag/manifests/{project_id}.json`

增量索引通过 manifest 对比文件 sha256、chunker_version、embedding_model 判断变化，未变化文件复用旧 chunks/vectors。

## QA 评估系统

独立服务，默认端口 8020：

```bash
export DASHSCOPE_API_KEY=sk-xxx
python -m src.main.python.evaluation_main --host 0.0.0.0 --port 8020
```

评估页面：`http://localhost:8020/evaluation`

支持 Gold QA、RAG 忠实度、弱监督质量三类评估，使用 MiniMax 大模型裁判。详见 README 历史版本。

## 批量采集

```bash
python -m src.main.python.eval.collect_rag_answer_results \
  --input docs/queries.txt \
  --output outputs/rag_answer_results.jsonl \
  --workers 4
```
