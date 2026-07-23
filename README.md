# Guga

Guga 是一个使用 Python 构建的长期记忆型 AI 聊天助手，支持本地模型部署和
DeepSeek API 调用，包含流式对话、Tool Calling、分层记忆、BGE-M3 + FAISS 检索、
角色隔离和 GPT-SoVITS 语音输出。

## 主要能力

- **流式对话**：支持在线 API、本地模型、多轮会话和 `Ctrl+C` 中断。
- **独立人格记忆**：不同 persona 的对话、长期记忆和检索索引彼此隔离。
- **分层记忆**：原始对话会逐步整理为事件、摘要与用户画像，供之后的相关对话使用。
- **自动整理**：默认每 10 个完整 turn，以及退出 session 时整理记忆。
- **语义检索**：BGE-M3 + FAISS 从相关记忆中挑选当前需要的背景。
- **可选评测**：提供隔离的 LongMemEval 评测入口。

## 记忆与人格

Guga 会保留原始对话，并定期将值得延续的内容整理为三类背景：正在发生或将发生的
事件、跨对话仍有价值的摘要，以及对用户偏好和习惯的工作性理解。不同 persona 的
记忆彼此隔离。

默认人格来自 `config/personas/default/SKILL.md`。Skill 同时决定日常对话的语气，
以及 Guga 在整理记忆时如何理解事件、形成自己的主观 reflection。reflection 是
Guga 的感受和判断，客观事件仍以用户实际说过的内容为准。

```text
原始对话
    ↓
每 10 个完整 turn / 退出 session
    ↓
事件、摘要、用户画像
    ↓
在相关的新对话中作为背景使用
```

## RAG 与上下文编排

正式语义检索使用：

```text
文本 / query
  → BAAI/bge-m3
  → 1024 维归一化向量
  → FAISS IndexFlatIP
  → 分层结果选择与 prompt 裁剪
```

- 长对话会切分为适合检索的片段；每轮只选取与当前话题相关的少量记忆。
- 事件优先于长期摘要，用户画像只用于理解用户，不覆盖客观事实。
- 每轮发送给模型的消息顺序如下：

```text
system message
  ├─ Persona Skill
  ├─ active semantic events
  ├─ archival memory
  ├─ derived event summaries
  ├─ raw evidence chunks
  ├─ guga user model
  └─ relevant documents / current rule
ChatHistory（按时间由旧到新，最近 45 个完整 turn 的原文）
本轮 user input
```

`ChatHistory` 超出 45 个完整 turn 时，从最旧的一组 user/assistant 消息开始裁剪；
检索记忆始终位于 system message 内，不插入短期原文历史。

`Semantic Events` 是当前事实的最高优先级来源；`Derived Event Summaries` 不是事实源；
`Raw Evidence` 用于核验原始说法；`Guga User Model` 只用于理解用户，不能覆盖客观事实。

共享文档放在 `data/documents/`。文档内容可共享读取，每个 persona 保持自己的对话
记忆与检索索引。

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

### DeepSeek API

```env
Guga_MODEL_PROVIDER=api
Guga_MODEL_ID=deepseek-v4-pro
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

表情标签会在送入 TTS 前从正文中过滤，因此不会被朗读出来。

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

## 可选：LongMemEval 评测

LongMemEval 用于评估长期记忆效果，运行数据与日常聊天数据隔离：

```text
data/benchmarks/longmemeval/runs/<run_id>/
```

评测使用专用 system prompt，不读取日常 persona 记忆。

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
- `--no-semantic`：关闭语义检索，适合轻量流程测试。

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

真实 API 的 Persona Skill 验收测试默认跳过。显式设置
`GUGA_RUN_LIVE_API_TESTS=1` 后可运行，测试会发起真实 API 请求并可能产生费用：

```powershell
$env:GUGA_RUN_LIVE_API_TESTS = "1"
python -m unittest discover -s test -p test_persona_skill_live_api.py
```

更多开发诊断工具位于 `scripts/` 与 `tools/`。
