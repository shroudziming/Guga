# Guga 人格 Skill 运行时接入设计

## 目标

将 `penguin-administrator-companion` 的人格 Skill 复制进 Guga 仓库，取代
`config/personas/default.json` 中的简单人格 prompt。精简后的完整 Skill 正文在每次
主对话请求中注入，并在现有低层记忆整理调用中作为 `guga_reflection` 的观察视角。

本次不引入 Self Model，不增加模型调用，不更换 `agent_id`，并继续使用
`data/memory/agents/default/` 中已有的全部记忆。

## 已确认的产品约束

- Guga 运行时始终启用该人格，不再要求用户显式说出角色名。
- 每次主对话都注入完整 Skill 正文，不按章节裁剪。
- 每批低层记忆整理也注入同一份完整 Skill 正文。
- `guga_reflection` 仍在当前批量整理中产生：通常每 10 个完整回合一次，未满批次可在
  session flush 时提前整理。
- Skill 只影响 `guga_reflection`，不得改变客观事件、时间、状态或来源证据。
- `[表情]` 标签保留为宿主控制协议，但不发送给 TTS；未来可交给 Live2D 适配器。
- Skill 内容变化不意味着角色身份变化，也不应导致已有记忆不可读。

## 仓库结构

```text
config/personas/
├── default.json
├── gentle.json
├── rational.json
└── default/
    ├── SKILL.md
    └── references/
        └── research/
```

`default.json` 改为通过相对路径引用 `default/SKILL.md`。`gentle` 和 `rational`
继续支持现有内联 `system_prompt`，避免无关迁移。

调研材料保留在 Skill 目录中用于来源审计，但不自动拼入运行时 prompt。每次模型调用
注入的是完整 `SKILL.md` 正文；YAML frontmatter 只用于加载和版本识别。

## Skill 加载

新增 `guga/persona/skill_loader.py`。加载器在进程启动时执行一次：

1. 将 `skill_path` 解析为 `config/personas/` 下的真实路径，并拒绝路径越界。
2. 解析仅含 `name` 和 `description` 的 YAML frontmatter。
3. 读取 frontmatter 后的完整 Markdown 正文，验证正文非空。
4. 从“允许的表情标签”协议中读取固定标签并验证无重复。
5. 对 Skill 文件内容计算 SHA-256 指纹。
6. 返回完整正文、标签集合、来源路径和指纹。

`PersonaManager` 在配置含 `skill_path` 时委托该加载器；否则保留现有内联 prompt
兼容路径。Skill 加载后缓存在 `Persona` 对象中，不在每轮对话时重新读取磁盘。

## Prompt 组装

### 主对话

主对话继续由 `MemoryManager.compose_system_prompt()` 组装：

```text
[Task Mode: Conversation]
[Persona Skill]
<完整 SKILL.md 正文>

<动态记忆、用户模型、相关文档和事实优先级规则>
```

Skill 位于稳定前缀，动态证据位于其后。这样既保持人格始终在场，也尽量为支持前缀
缓存的模型服务保留稳定输入。系统不假设供应商一定提供 prompt caching。

### 低层记忆整理

`MemoryManager` 调用 `consolidate_low_level_memory()` 时新增显式参数
`reflection_context`，其值为完整 Skill 正文。

整理 prompt 使用以下优先级：

```text
[Task Mode: Memory Reflection]
宿主 JSON schema 与客观事件规则
[Persona Skill]
<完整 SKILL.md 正文>
[Reflection Contract]
待整理对话与证据
```

Memory Reflection 模式下：

- Skill 的身份、心智模型、关系原则和价值张力只用于填写 `guga_reflection`。
- Skill 的聊天输出格式、表情标签、主动追问和直接对话要求不执行。
- `semantic_event_operations`、`event_summaries`、时间和来源字段只服从证据与宿主
  schema。
- benchmark 设置 `include_guga_reflection=false` 时不注入 Skill，保持评测隔离。

高层整理继续更新 archival memory 和 user model，不注入 Skill，也不新增调用。

## API 调用与成本

接入 Skill 不增加任何模型调用：

- 主对话仍使用原来的生成调用。
- 低层整理仍使用原来的结构化生成调用，只增加人格输入。
- 高层整理保持不变。
- 现有 tool calling、JSON 校验重试和批次失败重试行为保持不变。

代价仅是主对话与低层整理的输入 token 增加。为此必须精简 Skill 本身，而不是在
运行时选择章节。

## Skill 精简

当前 Skill 约 344 行、7479 字符。目标压缩到约 180–220 行、4000–5000 字符，
保留完整人格行为而去除宿主无关或重复内容。

### 删除或移出正文

- 删除“用户明确点名才激活”和“退出后恢复默认助手”；Guga 宿主始终启用人格。
- 将详细调研来源和生成方法移入现有 `references/research/`。
- 删除面向 Codex 的工具触发说明，改为只使用宿主实际提供的工具。
- 删除重复的语音说明；仅保留表情标签不是朗读正文。

### 合并和压缩

- 合并“身份”“自我认知”和“诚实边界”。
- 保留五个心智模型，但将重复的“一句话、怎么用、局限”压缩为规则表。
- 关系模型只保留长期互动原则，具体场景由状态机负责。
- 将 20 个表情标签压缩为分组映射，标签全集保持不变。
- 将回答工作流压缩为一条决策优先级，不重复状态机内容。
- 示例保留 3–4 个最能区分人格的场景。
- 将禁忌、反模式和核心张力合并为一个边界章节。

### 新增 Reflection 写作协议

Skill 新增明确的 Memory Reflection 条件分支：

- `appraisal`：Guga 如何评价事件对用户或双方关系的意义。
- `felt_response`：具体、克制、符合人格的主观感受。
- `relational_intent`：以后相处时希望保持或调整的互动方式。
- `interpretation_confidence`：对主观理解的把握程度。
- 证据不足时保持不确定，不为体现人格而强行制造感受。
- 不输出表情标签，不直接回复用户，不覆盖宿主 JSON schema。
- 不把反思写入客观事件字段。

## 人格身份与版本审计

现有 `agent_manifest.json` 将人格指纹作为启动硬校验，这会阻止修改 Skill 后继续读取
旧记忆。身份与版本应分离：

```text
agent_manifest.json       # 稳定身份，只决定记忆根目录归属
persona_revisions.jsonl   # Skill 版本审计
```

`agent_manifest.json` 升级到 schema 2，仅保存 `agent_id` 和创建时间。
`persona_revisions.jsonl` 在首次见到新指纹时追加来源、指纹和启用时间。

迁移规则：

- schema 1 且 `agent_id=default` 时原地升级，不移动或重写任何记忆文件。
- Skill 指纹变化只追加版本记录，不拒绝启动。
- `agent_id` 不同仍使用不同的 agent memory root。
- 非法 manifest、agent ID 不匹配和路径越界继续立即失败。

## 表情输出与未来 Live2D

新增流式 `PersonaOutputParser`，将模型输出拆成文本和表情事件：

```text
[happy]你好。
  ├─ ExpressionEvent("happy")
  └─ TextEvent("你好。")
```

解析器必须处理标签跨 chunk、连续多段、未知标签和流结束残留。只有 Skill 声明的标签
会成为表情事件；未知标签按普通文本处理。

当前消费者：

- 文本 CLI 输出纯正文，debug 可记录表情事件。
- VoiceChatRunner 将纯正文送入现有括号动作过滤器和 TTS。
- `expression_sink` 默认为空实现。

未来 Live2D 适配器只需实现 `expression_sink(tag)`，再将稳定标签映射到 Live2D
expression、motion 或模型参数。ChatSession、记忆系统和 TTS 客户端不依赖 Live2D。

原始 assistant 回复仍由 ChatSession 完整保存，保留表情控制协议用于审计和上下文一致性；
展示与朗读在外围解析。

## 错误处理

- Skill 不存在、frontmatter 非法、正文为空或路径越界：启动失败并指出文件与原因。
- Skill 未声明有效表情标签：启动失败，避免协议静默失效。
- 流式未知标签：作为普通文本透传，不吞掉用户可见内容。
- Reflection 结构化输出非法：沿用当前最多三次 JSON 校验重试。
- manifest 迁移写入失败：停止启动，不在身份状态不确定时继续写记忆。

## 测试

所有实现按测试优先顺序完成。

### Skill 加载

- 完整正文逐字进入主对话 prompt，不做章节裁剪。
- frontmatter 不进入运行时正文。
- 相对路径、路径越界、正文为空和标签协议错误得到覆盖。
- 内联 persona 兼容路径继续工作。

### Reflection

- 低层整理 prompt 包含完整 Skill 和 Memory Reflection 模式边界。
- Skill 只作用于 `guga_reflection`；客观事件规则保持更高优先级。
- `include_guga_reflection=false` 时不注入 Skill。
- 不新增模型调用。

### 记忆迁移

- 现有 schema 1 `default` manifest 原地升级。
- Skill 指纹变化后继续读取同一 memory root，并追加 revision。
- 不同 `agent_id` 仍隔离。

### 输出解析

- 完整标签、跨 chunk 标签、多段标签、未知标签和 flush。
- CLI 不显示控制标签。
- TTS 不朗读控制标签，但继续过滤括号动作。
- `expression_sink` 按流式顺序收到事件，为未来 Live2D 保持稳定接口。

### 验证命令

先运行新增和受影响测试，再运行完整测试：

```powershell
python -m unittest discover -s test -p "test_persona*.py"
python -m unittest discover -s test -p "test_memory_consolidation.py"
python -m unittest discover -s test -p "test_agent_memory_isolation.py"
python -m unittest discover -s test -p "test_voice_pipeline.py"
python -m unittest discover -s test
```

Skill 修改还需先记录无 Skill 约束时的基线输出，再用相同场景验证精简后 Skill 对
闲聊、互怼、严肃求助、事实边界和 Memory Reflection 的行为约束。

## 非目标

- 不实现 Self Model。
- 不让 Guga 自动修改 Skill。
- 不实现 Live2D 连接、动作映射或 UI。
- 不改变记忆批次大小。
- 不改变 semantic event、archival memory 或 user model 的数据职责。
- 不迁移 `gentle` 和 `rational` 到 Skill。
