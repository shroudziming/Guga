# Guga

Guga 是一个使用 Python 构建的长期记忆型 AI 聊天助手。项目支持本地
Qwen2.5-VL 和 OpenAI-compatible API，包含流式对话、tool calling、分层记忆、
BGE-M3 + FAISS 检索、角色隔离、GPT-SoVITS 语音输出，以及隔离的 LongMemEval
评测链路。

## 主要能力

- **流式对话**：支持在线 API、本地模型、多轮会话和 `Ctrl+C` 中断。
- **Agent 隔离**：不同 persona 使用独立 memory root、RAG 索引和 debug 报告。
- **分层记忆**：原始消息、语义事件、派生摘要、长期档案和用户模型职责分离。
- **批量整理**：默认每 10 个完整 user-assistant turn 和 session 结束时触发整理。
- **健壮状态机**：Stage 1/2 严格有序，支持结构化校验、重试、幂等提交和断点恢复。
- **正式语义检索**：BGE-M3 对完整 query 和记忆 chunk 生成 1024 维归一化向量，FAISS `IndexFlatIP` 执行检索，不使用词法相关性兜底。
- **LongMemEval**：评测数据、记忆、索引、debug、checkpoint 和结果均与日常数据隔离。

## 记忆架构

每个 agent 的数据保存在：

```text
data/memory/agents/<agent_id>/
```

主要持久文件：

| 文件 | 职责 |
| --- | --- |
| `sessions/*.jsonl` | 不可改写的原始 user/assistant 消息时间线 |
| `session_memories.jsonl` | 原始消息的可检索索引记录 |
| `semantic_events.jsonl` | 当前事实及事件生命周期的权威来源 |
| `event_summaries.jsonl` | 批次级派生摘要，仅用于压缩和定位 |
| `archival_memory.jsonl` | 跨事件的长期事实背景 |
| `guga_user_model.json` | 当前 agent 对用户的证据化工作性理解 |
| `consolidation_state.json` | 批次、stage、重试和恢复状态 |
| `rag/index/` | 当前 agent 独立的 FAISS 索引和模型元数据 |
| `debug_reports/` | 当前 agent 独立的调试记录 |

旧的 `data/memory/` 数据不会自动迁移，也不会被新的 agent root 回退读取。
persona 的身份发生本质变化时，应使用新的 `agent_id`，不要复用旧记忆目录。

### Semantic Event 生命周期

Stage 1 对事件只输出以下操作：

```text
create | update | replace | cancel | ignore
```

- `create`：创建新的 active event。
- `update`：保留 event ID，更新当前事实并追加证据消息。
- `replace`：旧 event 变为 `inactive/replaced`，并创建带 `replaces_event_id` 的新 active event。
- `cancel`：目标 event 变为 `inactive/cancelled`，不创建 successor。
- `ignore`：当前内容不形成语义事件。

inactive event 只保留在磁盘中用于审计、恢复和调试，不参与 Stage 1、Stage 2、
FAISS、词法检索或最终 prompt。引用 inactive event 的 derived summary、archival
memory 和 user model insight 同样会被来源有效性门禁排除。历史过程仍可从原始
session 消息中检索核验。

日常 agent 的 event 可包含角色化 `guga_reflection`，但该字段不参与事实真值、
时间字段或冲突处理。LongMemEval replay 会关闭 reflection 和 user model 更新。

### 两阶段批量整理

```text
原始消息逐条落盘
    ↓
每 10 个完整 turn / session end
    ↓
Stage 1: raw turns + 检索历史 → semantic events + derived summary
    ↓  JSON/schema/语义校验成功后提交
Stage 2: active events + derived summaries → archival memory + user model / no-op
```

Stage 2 不读取 raw session 或 pending turn 原文。任一 stage 失败时，后续 stage
和后续批次不会越过失败状态；待重试成功后才继续。状态文件会保留 active batch、
attempt、retry cycle 和幂等 commit key，进程重启后从未完成 stage 恢复。

## RAG 与上下文编排

正式语义检索使用：

```text
文本 / query
  → BAAI/bge-m3
  → 1024 维归一化向量
  → FAISS IndexFlatIP
  → 分层结果选择与 prompt 裁剪
```

- BGE-M3 或 FAISS 加载失败时直接报错，不会静默降级为 HashingEmbedder 或 Python 点积。
- 索引保存 embedding model 和 dimension；发现旧 Hashing/BGE-small 索引时会自动重建。
- 长原始消息按 chunk 写入索引，避免把整段 GDPR 等长文本直接塞入 prompt。
- 一般相关性排序只使用 BGE-M3 的 chunk 向量分数；active 状态、时间窗口和当前 turn 抑制仍作为确定性约束。
- Prompt 组装时，Semantic Events、Archival Memory、Derived Event Summaries、Raw Evidence 每层最多保留 3 条。
- 最终上下文按以下优先级组织：

```text
active semantic events
archival memory
derived event summaries
raw evidence chunks
guga user model
```

`Semantic Events` 是当前事实的最高优先级来源；`Derived Event Summaries` 不是事实源；
`Raw Evidence` 用于核验原始说法；`Guga User Model` 只用于理解用户，不能覆盖客观事实。

共享文档放在 `data/documents/`。文档内容可共享读取，但每个 agent 和 benchmark case
都会建立自己的索引，不共享对话记忆索引。

## 安装

推荐使用 Python 3.11：

```powershell
git clone https://github.com/shroudziming/Guga.git
Set-Location Guga
python -m pip install -r requirements.txt
```

首次启用语义检索时会下载 `BAAI/bge-m3`。本地 Qwen2.5-VL 还需要与机器匹配的
PyTorch/CUDA 环境；仅使用在线 API 时不需要加载本地聊天模型。

## 模型配置

CLI 会自动读取仓库根目录的 `.env`，已存在的系统环境变量优先。

### OpenAI-compatible API

```env
Guga_MODEL_PROVIDER=api
Guga_MODEL_ID=deepseek-chat
Guga_API_BASE_URL=https://api.deepseek.com
Guga_API_KEY=replace_with_your_key
Guga_API_TIMEOUT=90
```

也可以使用 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 作为兼容变量。

常用生成和调试配置：

```env
Guga_PERSONA=default
Guga_MAX_NEW_TOKENS=1024
Guga_TEMPERATURE=0.7
Guga_TOP_P=0.9
Guga_MEMORY_MAX_NEW_TOKENS=2048
Guga_MEMORY_USE_LLM_SUMMARY=1
Guga_MAX_TOOL_ROUNDS=3
Guga_DEBUG=1
```

### 本地模型

```env
Guga_MODEL_PROVIDER=local
Guga_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
Guga_CACHE_DIR=./models_cache
```

## 运行

文本 CLI：

```powershell
python -B src\basic_cli_chat.py
```

交互命令：

```text
/clear        清空当前 ChatSession 历史
/rag_rebuild  从当前 agent 记忆和 documents 重建索引
/exit         结束程序
```

可用 persona 位于 `config/personas/`，当前包含 `default`、`gentle` 和 `rational`。

### GPT-SoVITS 语音 CLI

语音入口要求已有可访问的 GPT-SoVITS HTTP 服务，并至少配置参考音频：

```env
GUGA_TTS_ENDPOINT=http://127.0.0.1:9880/tts
GUGA_TTS_REF_AUDIO_PATH=D:\path\to\reference.wav
GUGA_TTS_PROMPT_TEXT=参考音频对应文本
GUGA_TTS_TEXT_LANG=zh
GUGA_TTS_PROMPT_LANG=zh
```

```powershell
python -B src\voice_cli_chat.py
```

语音层位于 `guga/voice/`，不修改 `ChatSession` 的记忆和 RAG 边界。

## Tool Calling

模型支持 OpenAI-compatible `tool_calls` 时，Guga 会在同一 turn 内执行工具并把结果
返回模型。默认注册：

- `guga_parse_time`
- `guga_list_dir`
- `guga_read_file`

写文件和命令执行工具默认关闭：

```env
Guga_ENABLE_WRITE_TOOL=1
Guga_ENABLE_COMMAND_TOOL=1
```

工具调用最多执行 `Guga_MAX_TOOL_ROUNDS` 轮。

## LongMemEval Benchmark

LongMemEval 使用独立运行目录：

```text
data/benchmarks/longmemeval/runs/<run_id>/
```

每个 case 还会使用独立的 `cases/<case_id>/memory/`、checkpoint 和 debug 路径。
benchmark 使用专用 system prompt，不读取日常 persona 记忆。

轻量 raw 导入：

```powershell
python -B src\run_longmemeval_benchmark.py `
  --dataset D:\path\to\longmemeval.json `
  --run-id smoke_raw `
  --limit 1 `
  --ingest-mode raw
```

批量 replay 整理：

```powershell
python -B src\run_longmemeval_benchmark.py `
  --dataset D:\path\to\longmemeval.json `
  --run-id replay_001 `
  --limit 1 `
  --ingest-mode replay `
  --replay-finalize-every 10 `
  --progress-every-messages 50 `
  --debug
```

- `raw`：直接导入可检索 session memory，不运行两阶段整理。
- `replay`：按原始消息时间逐轮记录，每 N 个完整 turn 和 session end 整理。
- `--no-semantic`：关闭 BGE-M3/FAISS 检索且不启用词法兜底，仅用于轻量流程测试。
- 相同 `run-id` 会使用 case checkpoint 续跑；已完成 case 跳过，失败 stage 从保存状态恢复。
- benchmark 整理失败时不会继续最终问答，结果标记为 `consolidation_failed`。

评分：

```powershell
python -B src\score_longmemeval_results.py `
  --results data\benchmarks\longmemeval\runs\replay_001\results.jsonl
```

评分结果写入同目录的 `metrics.json` 和 `failures.jsonl`。

## 项目结构

```text
config/personas/        persona、agent_id 和 reflection_context
guga/chat/              ChatSession 与对话流程
guga/memory/            事件存储、两阶段整理、用户模型和恢复状态机
guga/rag/               BGE-M3 embedder、chunking、FAISS store 和检索 pipeline
guga/benchmark/         LongMemEval 数据加载、隔离 workspace 和运行编排
guga/models/            本地模型与 OpenAI-compatible API 适配
guga/voice/             GPT-SoVITS 客户端、句子缓冲和播放队列
src/                    CLI 与 benchmark 入口
scripts/                API、tool calling 和语音验证脚本
tools/rag_diagnostics/  独立 RAG 诊断工具
test/                   unittest 测试
```

## 测试

```powershell
python -B -m unittest discover -s test
```

真实记忆 API 小场景验证脚本位于 `scripts/live_memory_api_validation.py`；
独立于运行时检索的词法、BM25、BGE-M3 对照探针位于 `tools/rag_diagnostics/`。
