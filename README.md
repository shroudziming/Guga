# Guga

这里存放“聊天陪伴机器人”学习路线的代码与配置。

## 当前学习阶段

- 阶段一：本地模型推理 + 多轮聊天（进行中）
- 阶段二：人设与记忆（结构已预留）

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

