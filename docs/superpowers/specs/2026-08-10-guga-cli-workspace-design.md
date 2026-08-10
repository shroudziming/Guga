# Guga CLI 与会话级工作区设计

## 目标

为文本版 Guga 提供一个简单、可注释、可配置的 Windows 启动入口，并把文件和命令工具从固定项目根目录改为会话级工作区。

用户通过仓库根目录的 `guga_cli.ps1` 启动 Guga。启动配置合并到仓库已有的 `.env`，不再新增第二个 CLI 配置文件。默认走 API 路线，默认工作区为当前 Windows 用户桌面下的 `Guga` 文件夹。Guga 可以在已批准的任务中通过工具检查、切换或重置工作区；切换只对当前 Python 进程有效，重启后恢复默认工作区。

## 文件与职责

### `guga_cli.ps1`

根目录启动脚本负责：

1. 读取仓库根目录 `.env` 中的模型路线、模型名称、API 密钥、工具开关和工作区设置；不打印密钥。
2. 根据 `Guga_MODEL_PROVIDER` 选择 API 或本地模型；该值缺失时默认使用 `api`。
4. 解析 Windows 真实桌面路径，创建默认的 `Desktop/Guga` 工作区。
5. 启动 `src/basic_cli_chat.py`，并将默认工作区通过进程环境传入。
6. 当配置缺失、路线非法、API 密钥缺失或目录创建失败时，在启动模型前给出明确错误并返回非零退出码。

脚本不保存会话中的工作区切换，也不修改 `.env`。

### `.env`

现有 `.env` 继续作为唯一启动配置，其中既包含 API 密钥等未跟踪的敏感值，也包含 CLI 设置。新增或整理配置时必须保留用户已有密钥，不在终端、Trace 或提交中输出密钥。每项配置必须带中文注释，尤其明确布尔开关中 `1 = 允许/启用`、`0 = 禁止/关闭`。

```env
# 模型路线：api = 在线 API；local = 本地模型。
# 未设置时 guga_cli.ps1 默认使用 api。
Guga_MODEL_PROVIDER=api

# 当前路线使用的模型名称。
# API 示例：deepseek-v4-pro
# 本地示例：Qwen/Qwen2.5-VL-3B-Instruct
Guga_MODEL_ID=deepseek-v4-pro

# 本地模型缓存目录。仅 local 路线使用；相对路径以仓库根目录为基准。
Guga_CACHE_DIR=./models_cache

# 默认工作区：desktop 表示 Windows 桌面下的 Guga 文件夹。
Guga_CLI_DEFAULT_WORKSPACE=desktop

# 工作区工具是否允许创建不存在的目录：
# 1 = 允许创建；0 = 只允许切换到已存在目录。
Guga_CLI_ALLOW_CREATE_WORKSPACE=1

# 是否允许 Guga 写入文件：
# 1 = 允许；0 = 禁止。
# 写入仍只能发生在当前会话已确认的工作区内。
Guga_ENABLE_WRITE_TOOL=1

# 是否允许 Guga 执行命令：
# 1 = 允许；0 = 禁止。
# 即使设为 1，仍需展示任务计划、获得批准并确认当前工作区。
Guga_ENABLE_COMMAND_TOOL=1

# 是否显示调试日志：1 = 显示；0 = 关闭。
Guga_DEBUG=0
```

切换本地模型时修改同一文件中的两项：

```env
Guga_MODEL_PROVIDER=local
Guga_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
```

切回 API 时改回 `api` 与对应 API 模型名称。`.env` 继续被 Git 忽略，不得提交密钥、令牌或用户凭证。

## 会话级工作区模型

新增一个进程内 `WorkspaceContext`，由任务工具注册表共享。它保存：

- `default_root`：启动时解析出的 `Desktop/Guga`。
- `current_root`：当前会话工作区。
- `confirmed`：当前路径是否已由 `guga_workspace inspect` 确认。
- `allow_create`：是否允许创建不存在的目标目录。

`WorkspaceContext` 不写磁盘状态文件，不进入长期记忆，也不从 checkpoint 恢复。新的 Guga 进程总是以 `default_root` 开始。

现有 `guga_list_dir`、`guga_read_file`、`guga_write_file` 和 `guga_run_command` 不再捕获一个固定 `Path`，而是在每次执行时读取 `WorkspaceContext.current_root`。因此切换后无需重建工具注册表。

## `guga_workspace` 工具

工具参数：

```json
{
  "action": "inspect | set | reset",
  "path": "set 时可选的目标路径",
  "create_if_missing": "set 时可选的布尔值"
}
```

行为：

- `inspect`
  - 返回当前工作区和默认工作区。
  - 将当前路径标记为已确认。
- `set`
  - 只有当前工作区已先经过 `inspect` 才允许切换。
  - 绝对路径直接解析；相对路径相对于当前工作区解析。
  - `create_if_missing` 省略时默认为假。
  - 若目标不存在，仅当全局 `allow_create` 与调用参数 `create_if_missing` 都为真时创建。
  - 成功后返回旧路径和新路径，并把新路径标记为未确认。
- `reset`
  - 恢复到 `default_root`。
  - 恢复后标记为未确认，后续操作前必须再次 `inspect`。

路径解析后必须是目录。文件路径、空路径和无法解析的路径返回结构化错误，不改变当前工作区。

## 计划、审批与执行约束

普通聊天仍只拥有时间解析工具。`guga_workspace` 和其他操作工具只能进入 LangGraph 任务运行时。

任务规划控制提示词要求所有涉及文件或命令的计划先确认工作区：

```text
1. 调用 guga_workspace inspect 确认当前工作区
2. 如有需要，调用 guga_workspace set 切换工作区
3. 切换后再次调用 guga_workspace inspect
4. 执行列目录、读写文件或命令
5. 验证结果
```

运行时不只依赖提示词：

- 当 `WorkspaceContext.confirmed` 为假时，操作工具直接拒绝执行。
- `set` 和 `reset` 后自动失效确认状态。
- 计划修订时自动失效确认状态，因此修订后的新计划必须重新确认。
- 未批准前仍不执行包括 `inspect` 在内的任何工具。
- 工作区工具调用、旧路径、新路径、操作结果和计划修订写入现有任务 Trace。

工作区确认属于真实工具调用，但不会增加后续业务步骤的重试次数；它拥有独立计划步骤和独立执行 ID。

## 模型路线

### API

当 `Guga_MODEL_PROVIDER=api` 时：

- 使用 `.env` 中的 `Guga_MODEL_ID`；缺失时默认 `deepseek-v4-pro`。
- 要求 `.env` 已提供 `Guga_API_KEY` 或兼容的 `OPENAI_API_KEY`。
- 使用现有原生 tool calling 路线。

### 本地模型

当 `Guga_MODEL_PROVIDER=local` 时：

- 使用 `.env` 中的 `Guga_MODEL_ID`；缺失时默认 `Qwen/Qwen2.5-VL-3B-Instruct`。
- 将相对 `Guga_CACHE_DIR` 解析为仓库根目录下的绝对路径。
- 使用现有结构化 JSON action 回退路线。

路线值不是 `api` 或 `local` 时拒绝启动，不静默回退。

## 错误处理

- 默认工作区无法创建：启动失败，不回退到仓库根目录。
- API 路线缺少密钥：启动失败，不加载本地模型。
- 本地模型缓存不存在：允许现有模型加载器按原行为处理下载或报错。
- 未确认工作区就执行操作工具：返回 `ok=false`，说明必须先调用 `guga_workspace inspect`。
- 切换失败：保留旧工作区与原确认状态。
- Guga 退出：关闭 SQLite checkpoint；不保存当前工作区。

## 测试与验收

自动测试使用临时目录替代真实桌面，不写用户文件：

1. `.env` 配置默认路线为 API，且注释明确 `0/1` 含义。
2. 启动配置能正确选择 API 与本地模型变量。
3. 默认工作区不存在时自动创建。
4. 新会话从默认工作区开始。
5. 未 `inspect` 时，列目录、读取、写入和命令均被拒绝。
6. `inspect` 后操作工具使用当前工作区。
7. `set` 后必须重新 `inspect`，之后所有工具使用新工作区。
8. `reset` 后恢复默认路径并要求重新确认。
9. 计划修订使工作区确认失效并重新等待批准。
10. 新进程不继承上一进程切换的工作区。
11. 普通聊天仍只有时间解析工具。
12. 现有 LangGraph、记忆、人格、工具和 LongMemEval 测试保持通过。

手工验收命令为：

```powershell
.\guga_cli.ps1
```

启动后，Guga 默认工作区应显示为 Windows 桌面下的 `Guga` 文件夹；API 路线为默认值，文件写入与命令执行默认允许，但所有操作仍受计划批准和工作区确认约束。
