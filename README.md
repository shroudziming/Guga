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
- **审批式智能体任务**：使用 LangGraph 展示计划、等待批准、执行工具、验证并持久化恢复。
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

`guga_cli.ps1` 会读取两类互相隔离的配置：

- 仓库根目录的 `.env` 被 Git 忽略，只保存 API 密钥、base URL 等私有连接信息；已存在的系统环境变量优先。
- `config/guga_cli.env` 被 Git 跟踪，保存模型路线、模型名称、工具开关、调试和默认工作区等可共享设置。

启动配置使用固定键白名单，不能覆盖 `.env` 中的 API 密钥。

### DeepSeek API

```env
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
```

### 本地模型

默认使用 API 路线。要使用 `config/guga_cli.env` 中配置的本地模型，只需把
`Guga_CLI_MODEL_ROUTE=api` 改为 `Guga_CLI_MODEL_ROUTE=local`。本地模型名称和缓存目录也在
该启动配置中维护。

## 运行

文本 CLI：

```powershell
.\guga_cli.ps1
```

启动器会解析当前 Windows 用户的真实桌面路径，并创建默认工作区 `Desktop\Guga`。

交互命令：

```text
/task <任务>       创建任务并展示待批准计划
/tasks             列出当前 persona 的未结束任务
/resume <task_id>  选择重启后要恢复的任务
/approve           批准当前版本计划并持续执行
/reject            拒绝当前版本计划
/clear             清空当前 ChatSession 历史
/rag_rebuild       从当前 agent 记忆和 documents 重建索引
/exit              结束程序
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

文本 CLI 的普通聊天只注册时间解析工具 `guga_parse_time`。文件、写入和命令工具只允许
通过 `/task` 启动的任务运行时执行。模型支持 OpenAI-compatible `tool_calls` 时，任务
运行时使用原生工具调用；本地模型使用经过同样名称和参数校验的 JSON action 回退。

任务工具包括：

- `guga_parse_time`
- `guga_workspace`（`inspect`、`set`、`reset`）
- `guga_list_dir`
- `guga_read_file`
- `guga_write_file`
- `guga_run_command`

写文件和命令执行工具在 `config/guga_cli.env` 中默认允许：

```env
Guga_ENABLE_WRITE_TOOL=1
Guga_ENABLE_COMMAND_TOOL=1
```

允许开关不等于立即执行：任务仍需先展示计划并获得批准，随后调用
`guga_workspace inspect` 确认当前工作区。`set` 可切换工作区，`reset` 可恢复默认目录；
切换或重置后必须再次 `inspect`。这些变化只在当前 Guga 进程中有效，退出或重启后恢复
默认的 `Desktop\Guga`，不会写入启动配置、checkpoint 或长期记忆。

每个计划步骤最多进行三次真实工具执行。模型输出的 JSON/schema 修复最多尝试三次，
但不计入步骤执行次数。

## LangGraph 任务运行时

任务状态流为：

```text
planning → awaiting_approval → executing
    ↑                              ├─ completed
    └──── revised_plan ←───────────┤
                                   ├─ failed
                                   └─ blocked
```

- 未批准前不会执行工具；修订计划会再次等待 `/approve`。
- 每个步骤包含目标、预期结果、验证方法和允许工具，计划外工具会要求修订计划。
- 第三次执行仍与预期不符时进入 `failed`；连续三次无法生成有效内部协议时进入 `blocked`。
- 任务开始时只冻结一次相关记忆、用户模型和文档上下文，执行期间不会因记忆变化漂移。
- SQLite checkpoint 位于 `data/agent_runs/<agent_id>/checkpoints.sqlite3`。
- 可读 trace 位于 `data/agent_runs/<agent_id>/<task_id>/trace.jsonl`，逻辑引用为
  `agent-run://<agent_id>/<task_id>/trace.jsonl`。
- 恢复时复用已有完成记录；只有开始记录时先执行状态检测，无法判断则进入 `blocked`
  等待人工检查，不盲目重复原调用。
- 长期记忆的 `task_outcomes.jsonl` 只保存清理后的终态摘要与 trace 引用，不混入原始
  工具过程或用户语义事件。

CLI 使用 LangGraph 的 `updates` 与 `custom` stream 显示步骤、工具名、尝试次数和验证
结论，不显示原始 JSON、完整参数或完整工具输出。当前工具输出截断策略保持不变；完整
artifact 归档不在本次实现范围内。

### 智能体能力基准

以下命令使用当前配置的 API 或本地模型运行六项隔离能力测试：

```powershell
python -B src\run_agent_task_benchmark.py --run-id agent_smoke_001
```

case 覆盖读取文件、发现未知文件、确定性 Python 命令、读取—执行—验证、预期命令失败
恢复，以及写入后重新读取。成功与否由模型外部的确定性 verifier 判断。输出位于：

```text
data/benchmarks/agent_tasks/runs/<run_id>/
├─ results.jsonl
├─ metrics.json
└─ cases/<case_id>/trace.jsonl
```

指标包括通过率、工具调用数、重试数、计划修订数、耗时和失败原因。基准会在运行期间
临时开启写入与命令工具，并在结束后恢复原环境变量。

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
guga/agent/             LangGraph 计划、审批、执行、恢复和 trace
guga/memory/            事件存储、两阶段整理、用户模型和恢复状态机
guga/rag/               BGE-M3 embedder、chunking、FAISS store 和检索 pipeline
guga/benchmark/         LongMemEval 与智能体能力基准
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
