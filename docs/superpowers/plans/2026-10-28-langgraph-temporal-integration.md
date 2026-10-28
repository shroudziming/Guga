# LangGraph 与时间记忆集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 temporal 时间记忆链路及当前主线修复合入 LangGraph Agent 分支，同时保留任务运行时、动态工作区和双方测试契约。

**Architecture:** 在 `feat/langgraph-agent-runtime` 上创建来自 `feat/temporal-memory-retrieval` 的标准 merge commit。冲突解决以 temporal 的会话/记忆时间链路和 LangGraph 的工具/任务骨架为主，通过小型交叉测试固定两套能力的组合边界。

**Tech Stack:** Python、unittest/pytest、LangGraph、Git merge

**Spec:** `docs/superpowers/specs/2026-10-28-langgraph-temporal-integration-design.md`

## Global Constraints

- 不重写分支历史，不使用 rebase 或强推。
- 不修改或提交用户现有的 `docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md` 改动。
- 普通对话不得获得工作区写文件和执行命令等操作型工具。
- 时间工具必须按 ChatSession 绑定对应的 MemoryManager，不能被复用 registry 串线。
- 任务终态记忆必须继续幂等写入，且不能伪装成用户语义事件。

## Review Focus

- 同一个基础 registry 被多个 ChatSession 复用时，时间搜索仍调用各自的 MemoryManager；由 Task 2 的交叉测试覆盖。
- 普通对话 registry 仍隔离工作区操作型工具，但允许时间搜索；由 Task 2 的 registry 测试覆盖。
- Agent 任务终态写入与 temporal 时间字段/检索初始化可以同时工作；由 Task 3 的桥接测试覆盖。
- CLI 同时支持 `/task` 路径和 Windows 安全输入函数，正常退出时完成记忆收尾；由 Task 4 的 CLI 测试覆盖。
- 流式调用取消、空回复重试或工具异常后不会遗留活动时间工具上下文；由 temporal 原有工具调用测试覆盖。

---

### Task 1: 建立基线并启动合并

**Files:**
- Preserve: `docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md`
- Merge source: `feat/temporal-memory-retrieval`

**Interfaces:**
- Consumes: 当前 `feat/langgraph-agent-runtime` 和本地 `feat/temporal-memory-retrieval`
- Produces: 保留 merge 状态但尚未提交的工作树，仅预期 6 个共享文件冲突

- [ ] **Step 1: 记录用户改动与分支状态**

Run: `git status --short --branch` and `git diff -- docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md`

Expected: 只有该设计文档为预先存在的未提交改动，且当前分支是 `feat/langgraph-agent-runtime`。

- [ ] **Step 2: 运行 LangGraph 侧基线测试**

Run: `python -m pytest -q -p no:cacheprovider test/test_agent_task_graph.py test/test_agent_runner.py test/test_agent_task_memory.py test/test_tool_calling.py test/test_basic_cli_agent_wiring.py`

Expected: PASS；如有既有失败，先记录且不得误称为合并引入。

- [ ] **Step 3: 启动非快进合并但暂不提交**

Run: `git merge --no-ff --no-commit feat/temporal-memory-retrieval`

Expected: Git 保留 merge 状态，并报告设计中列出的 6 个内容冲突。

### Task 2: 整合工具 registry 与会话时间工具

**Files:**
- Modify: `guga/tools.py`
- Modify: `guga/chat/session.py`
- Test: `test/test_tool_calling.py`

**Interfaces:**
- Consumes: `WorkspaceContext`、`ToolRegistry`、`MemoryManager.search_time_context(...)`
- Produces: `conversation_tool_registry()`、`default_tool_registry(...)`、`memory_time_search_tool(handler)`，以及每个 ChatSession 独立绑定的时间搜索工具

- [ ] **Step 1: 合并双方测试并增加交叉失败测试**

在 `ToolCallingTest` 增加 `test_conversation_registry_binds_time_search_without_exposing_operational_tools`：创建带 workspace 的基础 registry 和两个不同 MemoryManager/ChatSession，断言普通对话 schema 不含写文件/命令工具、时间工具分别返回对应 manager 的结果。

- [ ] **Step 2: 验证测试因组合能力缺失而失败**

Run: `python -m pytest -q -p no:cacheprovider test/test_tool_calling.py`

Expected: FAIL，原因是冲突尚未解决或 registry 尚不能同时满足时间绑定与工具隔离。

- [ ] **Step 3: 以 LangGraph 工具骨架整合 temporal 能力**

保留 `WorkspaceContext`、名称过滤、参数模式校验和操作工具；实现 registry 的安全复制，加入 `memory_time_search_tool(handler)`，并让 ChatSession 在副本上绑定自身 manager。会话生成循环采用 temporal 的轮次追踪、重试和清理逻辑。

- [ ] **Step 4: 验证工具与会话测试通过**

Run: `python -m pytest -q -p no:cacheprovider test/test_tool_calling.py test/test_chat_session_rag_flow.py`

Expected: PASS。

### Task 3: 整合任务终态记忆与 temporal MemoryManager

**Files:**
- Modify: `guga/memory/manager.py`
- Test: `test/test_agent_task_memory.py`
- Test: `test/test_temporal_retrieval.py`
- Test: `test/test_memory_consolidation.py`

**Interfaces:**
- Consumes: `TaskOutcome`、`TaskOutcomeStore`、temporal retrieval records and consolidation state
- Produces: `MemoryManager.record_task_outcome(outcome) -> bool` 与 `MemoryManager.search_time_context(...)` 在同一实例共存

- [ ] **Step 1: 增加交叉失败测试**

在 `TaskOutcomeMemoryBridgeTest` 增加 `test_temporal_manager_keeps_task_outcome_bridge`：同一临时 MemoryManager 先幂等记录任务终态，再执行空数据时间搜索，断言终态文件保持一条且时间搜索返回合法空结果。

- [ ] **Step 2: 验证测试因桥接能力缺失而失败**

Run: `python -m pytest -q -p no:cacheprovider test/test_agent_task_memory.py test/test_temporal_retrieval.py`

Expected: FAIL，原因是 temporal 主体尚未重新接入任务终态存储或接口不兼容。

- [ ] **Step 3: 以 temporal MemoryManager 为主体恢复任务终态桥接**

保留 temporal 的来源规范化、时间区间、巩固水位和新 `compose_system_prompt` 上下文；重新加入 `TaskOutcomeStore` 初始化、`record_task_outcome` 和任务模式 prompt 支持，不恢复被 temporal 替代的旧日期查询分支。

- [ ] **Step 4: 验证记忆测试通过**

Run: `python -m pytest -q -p no:cacheprovider test/test_agent_task_memory.py test/test_temporal_retrieval.py test/test_memory_consolidation.py test/test_memory_manager.py`

Expected: PASS。

### Task 4: 整合 CLI、文档与剩余冲突

**Files:**
- Modify: `src/basic_cli_chat.py`
- Modify: `README.md`
- Test: `test/test_basic_cli_agent_wiring.py`
- Test: `test/test_cli_input.py`
- Test: `test/test_cli_shutdown.py`

**Interfaces:**
- Consumes: `TaskCommandController`、`AgentTaskRunner`、`read_user_input(prompt)`、ChatSession 退出收尾接口
- Produces: 同时支持普通聊天、任务命令、动态工作区和安全输入的 CLI

- [ ] **Step 1: 合并 CLI 测试并补充交叉失败测试**

在 `test/test_basic_cli_agent_wiring.py` 增加 wiring 断言：入口同时引用 `read_user_input` 和 `TaskCommandController`，且普通聊天仍使用 conversation registry、任务运行器使用 operational registry。

- [ ] **Step 2: 验证测试因冲突入口缺失而失败**

Run: `python -m pytest -q -p no:cacheprovider test/test_basic_cli_agent_wiring.py test/test_cli_input.py test/test_cli_shutdown.py`

Expected: FAIL，原因是 CLI 冲突尚未同时保留两侧 wiring。

- [ ] **Step 3: 以 LangGraph CLI 为主体接入 temporal 输入与收尾**

保留任务命令、模型适配器、runner 和 workspace 注入；将所有用户输入改为 `read_user_input`，保留 temporal 正常退出记忆巩固。README 分区保留两侧使用说明并删除重复内容。

- [ ] **Step 4: 验证 CLI 测试通过**

Run: `python -m pytest -q -p no:cacheprovider test/test_basic_cli_agent_wiring.py test/test_cli_input.py test/test_cli_shutdown.py test/test_agent_cli.py`

Expected: PASS。

### Task 5: 全量验证、记录冲突选择并完成合并

**Files:**
- Verify: all merged repository files
- Preserve: `docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md`

**Interfaces:**
- Consumes: Tasks 1-4 的已解决 merge index
- Produces: 可推送的标准 merge commit，以及逐文件冲突决策说明

- [ ] **Step 1: 检查索引与冲突标记**

Run: `git diff --check`; `git diff --name-only --diff-filter=U`; `rg -n "^(<<<<<<<|=======|>>>>>>>)" -g "*.py" -g "*.md"`

Expected: 无未合并路径、无冲突标记、无 whitespace error。

- [ ] **Step 2: 运行完整测试套件**

Run: `python -m pytest -q -p no:cacheprovider`

Expected: 0 failures；任何既有或环境失败必须逐项报告。

- [ ] **Step 3: 审核暂存范围和用户文件保护**

Run: `git status --short`; `git diff --cached --stat`; `git diff -- docs/superpowers/specs/2026-08-10-guga-cli-workspace-design.md`

Expected: 用户文件仍是未暂存修改且不在 merge index；其他变更均属于两分支集成。

- [ ] **Step 4: 创建并推送 merge commit**

Run: `git commit -m "merge(memory):整合时间记忆与任务运行时"` then `git push origin feat/langgraph-agent-runtime`

Expected: commit 和 push 成功，远端分支指向新 merge commit。

- [ ] **Step 5: 输出冲突决策报告**

逐项说明 `README.md`、`guga/chat/session.py`、`guga/memory/manager.py`、`guga/tools.py`、`src/basic_cli_chat.py`、`test/test_tool_calling.py` 的冲突原因、主体采用侧、保留的另一侧能力和验证证据。

