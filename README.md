# Guga

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

