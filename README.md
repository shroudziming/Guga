# Guga

## 当前 RAG / FAISS 状态

- 本地 RAG 流程同时支持对话记忆和文档片段检索。
- 向量索引由 memory JSONL 文件和支持的文档文件重建，并持久化到 `data/memory/rag/index/`。
- FAISS 检索现在保留全局索引，并按 `source_type` 维护分类型索引；`memory` 和 `document` 查询会直接使用对应 FAISS 索引，不再因为来源过滤退回 Python 点积遍历。
- `.codex/` 属于本地 Codex 工作区状态，不应由 Git 追踪。

这里存放“聊天陪伴机器人”学习路线的代码与配置。

## 当前学习阶段

- 阶段一：本地模型推理 + 多轮聊天（已完成基础版）
- 阶段二：人设与记忆（进行中，已接入本地 RAG 检索）

## 近期进度（2026-04-19）

- 已完成：CLI 多轮聊天（含流式输出与中断）
- 已完成：Persona 加载与切换（`default/gentle/rational`）
- 已完成：会话消息落盘与归档记忆写回
- 已完成：本地 RAG 检索闭环（memory + documents，语义 + 词法混合检索）
- 已完成：OpenAI 兼容 API 模型接入（使用中转站 `OPENAI_BASE_URL/OPENAI_API_KEY`），本地模型不够智能，无法评估RAG效果
- 进行中：记忆质量提升（召回稳定性、写回策略、评估与可观测性）

## 目录说明

- `Guga/guga/models/`：模型加载与生成
- `Guga/guga/chat/`：会话与对话历史
- `Guga/guga/persona/`：人设加载与切换
- `Guga/guga/memory/`：短期/长期记忆模块（阶段二预留）
- `Guga/guga/utils/`：通用工具
- `config/personas/`：人设配置文件
- `src/basic_cli_chat.py`：第4-5课多轮 CLI 聊天入口（含流式输出）

## 第4课运行方式

在工作区根目录执行：

`python Guga/src/basic_cli_chat.py`

可选环境变量：

- `Guga_MODEL_ID`：模型 ID（默认 `Qwen/Qwen2.5-VL-3B-Instruct`）
- `Guga_CACHE_DIR`：模型缓存目录
- `Guga_PERSONA`：人设名（默认 `default`）

## 使用在线 API 模型（实验）

支持 OpenAI 兼容接口，配置以所使用中转站为准

- `Guga_MODEL_PROVIDER` 不设或设为 `local` 时，仍使用本地模型。
- API 路径已支持流式输出（当中转站支持 SSE `stream=true` 时生效）。

### .env

CLI 启动时会自动读取 `Guga/.env`（若环境变量已存在，则以现有环境变量优先）。

示例：

```env
Guga_MODEL_PROVIDER=api
Guga_MODEL_ID=replace_with_your_model_id
OPENAI_BASE_URL=replace_with_your_api_url
OPENAI_API_KEY=replace_with_your_token
Guga_API_TIMEOUT=90
```

可选生成参数：

```env
Guga_MAX_NEW_TOKENS=1024
Guga_TEMPERATURE=0.7
Guga_TOP_P=0.9
Guga_MEMORY_MAX_NEW_TOKENS=512
```

MemoryBank 的事件摘要、用户画像和长期记忆抽取会复用当前聊天模型。使用 DeepSeek 等 OpenAI-compatible API 时，可按如下配置：

```env
Guga_MODEL_PROVIDER=api
Guga_MODEL_ID=deepseek-chat
Guga_API_BASE_URL=https://api.deepseek.com
Guga_API_KEY=replace_with_your_deepseek_key
Guga_API_TIMEOUT=90
```

默认策略：只要 `MemoryManager` 收到当前模型对象，就会启用 LLM 记忆总结；如需关闭并使用规则 fallback，可设置：

```env
Guga_MEMORY_USE_LLM_SUMMARY=0
```

### Tool calling

When the active model supports OpenAI-compatible `tool_calls`, Guga can expose a
small local tool registry to the model during generation. Tool results are fed
back into the same turn so the model can continue answering naturally.

Enabled by default:

- `guga_parse_time`: parse explicit and relative time expressions using Beijing
  time.
- `guga_list_dir`: list files under the Guga project root.
- `guga_read_file`: read UTF-8 text files under the Guga project root.

Registered but disabled by default:

- `guga_write_file`: enable with `Guga_ENABLE_WRITE_TOOL=1`.
- `guga_run_command`: enable with `Guga_ENABLE_COMMAND_TOOL=1`.

Tool loop limit:

```text
Guga_MAX_TOOL_ROUNDS=3
```

## LongMemEval benchmark

Guga 提供一条与日常聊天隔离的 LongMemEval 测评入口：

```text
python src/run_longmemeval_benchmark.py --dataset path/to/longmemeval.jsonl --limit 10 --no-semantic
```

- 测评状态写入 `data/benchmarks/longmemeval/runs/<run_id>/`。
- 每个 case 使用独立 `cases/<case_id>/memory/`，避免不同题目之间互相污染。
- debug 报告写入该 run 下的 `debug_reports/`，不会使用日常 `data/memory/debug_reports/`。
- 问答阶段使用 LongMemEval 专用 system prompt，不复用日常 persona 配置。
- `--limit` 适合先做小样本烟测；去掉 `--no-semantic` 后会为 benchmark 专用 memory/documents 建立独立 RAG 索引。
- `--ingest-mode raw` 会把历史直接导入为可检索 memory；`--ingest-mode replay` 会逐轮调用现有记忆整理链路，更贴近日常聊天，但 API 成本和耗时会显著增加。

对 `results.jsonl` 做规则化评分：

```text
python src/score_longmemeval_results.py --results data/benchmarks/longmemeval/runs/<run_id>/results.jsonl
```

评分脚本会在同目录写入 `metrics.json` 和 `failures.jsonl`。

