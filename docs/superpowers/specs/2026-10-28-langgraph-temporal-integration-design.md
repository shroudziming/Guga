# LangGraph 与时间记忆集成设计

## 意图

在 `feat/langgraph-agent-runtime` 上合入 `feat/temporal-memory-retrieval`，同时获得当前 `main` 的 12 个后续提交和 temporal 分支的 31 个时间记忆提交，并保留 LangGraph 任务运行时、动态工作区与 CLI 能力。

## 目标

- 使用标准 merge commit 保留两条分支的真实历史。
- 同时保留任务运行时和时间记忆检索、巩固、轮次追踪能力。
- 保留 Windows CLI 输入修复。
- 对每个冲突文件记录冲突原因、采用侧和最终整合方式。
- 用双方原有测试和新增交叉集成测试证明两套能力可以共存。

## 非目标

- 不重写或压缩已有提交历史。
- 不借合并机会重构无关代码。
- 不修改或提交 worktree 中现有的 `2026-08-10-guga-cli-workspace-design.md` 用户改动。
- 不改变现有工具、记忆或任务运行时的对外行为，除非为兼容双方能力所必需。

## 约束

- 集成目标分支是 `feat/langgraph-agent-runtime`。
- 集成来源分支是 `feat/temporal-memory-retrieval`；它已包含当前 `main`，无需再次单独合并 `main`。
- 合并前后均需明确检查用户未提交文件仍被保留且未进入提交。
- 验证通过后才允许完成 merge commit 和推送。

## 方案比较

1. **标准 merge commit（采用）**：历史完整，不需强推，能清楚显示两条功能线的汇合点；代价是需要人工解决共享核心文件冲突。
2. **将 LangGraph rebase 到 temporal**：历史线性，但会重写 54 个已推送提交并需要强推，风险不合适。
3. **选择性 cherry-pick**：可缩小表面改动，但 temporal 的 31 个实现提交依赖紧密，容易遗漏迁移、测试或时间证据链路。

## 冲突决策

| 文件 | 主体采用 | 必须保留的另一侧能力 | 整合原则 |
|---|---|---|---|
| `README.md` | 双方合并 | LangGraph 使用与 temporal 时间记忆说明 | 按功能分区保留，不用整文件选择一侧 |
| `guga/chat/session.py` | temporal | LangGraph 的会话工具隔离与任务运行时兼容入口 | 以 temporal 的轮次追踪、时间工具绑定和重试循环为主体，保留 conversation registry 的安全默认值 |
| `guga/memory/manager.py` | temporal | LangGraph 的 `TaskOutcome`、`TaskOutcomeStore` 和 `record_task_outcome` | 以 temporal 的时间检索和巩固模型为主体，重新接入任务终态记忆，不恢复已被 temporal 替换的旧时间查询实现 |
| `guga/tools.py` | LangGraph | temporal 的 registry 复制、`memory_time_search` 工具和会话绑定 | 以动态 `WorkspaceContext`、参数校验和工具范围过滤为骨架，加入时间工具而不让操作型工具泄漏到普通对话 |
| `src/basic_cli_chat.py` | LangGraph | temporal 的 `read_user_input`、轮次时间链路与正常退出收尾 | 保留任务命令和 workspace 注入，同时使用新输入函数并确保退出时完成记忆收尾 |
| `test/test_tool_calling.py` | 双方合并 | 双方全部独有测试及交叉场景 | 不删除任一侧契约；增加时间工具与动态工作区共存的测试 |

## 验证策略

- 合并前记录 LangGraph 相关测试基线。
- 冲突解决时先让交叉集成测试暴露缺失行为，再补最小整合代码。
- 运行 CLI、工具调用、任务运行时、时间检索、记忆巩固相关测试。
- 最后运行完整测试套件，并检查没有冲突标记、未合并索引或意外暂存用户文件。

