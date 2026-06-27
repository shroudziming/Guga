# 时间线事实层实现与实测报告

日期：2026-06-27

## 背景

原有 `prepare_context()` 最终能进入 prompt 的长期记忆主要来自：

- `archival_memory.jsonl`
- `event_summaries.jsonl`
- `session_memories.jsonl`
- `profile.json` 中的用户画像

这些结构适合“摘要”和“对话片段”，但不适合精确回答“某天我要做什么”。原因是 daily summary 是压缩后的自然语言摘要，容易丢细节；session turn 又太原始，容易把当前 turn 或低价值问句一起召回。

本次实现新增 `timeline_facts.jsonl`，定位是“补充事实层”，不是替代 summary。它存储独立、可检索、可失效的时间事实，例如：

```json
{
  "type": "timeline_fact",
  "subject": "user",
  "predicate": "has_time_bound_plan",
  "object": "要整理周报",
  "created_at": "2026-06-27T22:18:38+08:00",
  "valid_from": "2026-07-05T00:00:00+08:00",
  "semantic_day": "2026-07-05",
  "source_session_id": "sess_d55a8fc639ee",
  "source_message_ids": ["msg_6e470296f8"]
}
```

## 实现做法

新增 `guga/memory/timeline_facts.py`，提供 `TimelineFactStore`：

- 只从带语义时间的计划类用户输入中抽取事实。
- 使用现有 `extract_semantic_time()` 统一解析日期。
- `created_at/updated_at` 表示写入记忆的北京时间。
- `valid_from/valid_at/semantic_day` 表示语义上的事实时间。
- 记录 `source_session_id/source_message_ids`，用于追溯和去重。

接入 `MemoryManager`：

- 在 `record_user_message()` 阶段同步写入确定性事实，避免等待后台 LLM summary。
- 在 `prepare_context()` 的候选集合中加入 `timeline_facts.jsonl`。
- 当前 turn 产生的 fact 仍参与 `current_turn_factor=0.2` 弱化，不会压过历史事实。
- 日期命中时 `timeline_fact` 使用 `temporal_rule=date_match_timeline_fact`，相对普通 turn 有更高优先级。
- 同源 `timeline_fact` 与 `event_summary` 同时命中时，日期 query 下保留事实、去掉同源 summary，避免 prompt 重复。

接入 RAG pipeline：

- 全量重建索引时增加 `timeline_facts.jsonl`。
- 后台 finalize 仍可把已写入的事实增量加入语义索引。

## 真实测试结果

运行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
@'
真实时间线最终测试：我在2026年7月5日要整理周报，请你记住。
你记得我2026年7月5日要做什么吗？
/exit
'@ | python src\basic_cli_chat.py
```

真实输出中，第二轮回答命中：

```text
2026年7月5日 — 你要整理周报！
```

debug report：

```text
data/memory/debug_reports/2026.6.27.22.18.sess_d55a8fc639ee.log
```

关键证据：

- 第一轮用户消息入库后立刻写入事实：`timeline_fact_added fact_id=fact_d42c8f5880 day=2026-07-05`
- 第一轮检索中当前 turn fact 被弱化：`score=0.306`, `is_current_turn=true`
- 第二轮追问时 top1 为事实层：`hit_ids=['fact_d42c8f5880', ...]`
- 第二轮事实分数高于原始 turn：`timeline_fact score=1.0299`, `conversation_turn score=0.7699`
- 分数组件显示时间规则生效：`temporal_rule=date_match_timeline_fact`
- 当前查询 turn 被弱化：`current_turn_factor=0.2`, `score=0.254`

`timeline_facts.jsonl` 中新增的 active fact：

```text
fact_d42c8f5880
semantic_day=2026-07-05
valid_from=2026-07-05T00:00:00+08:00
object=真实时间线最终测试：要整理周报
status=active
```

## 修正过的问题

第一次真实测试暴露出一个污染问题：

```text
你记得我2026年7月3日要做什么吗？
```

该查询句曾被误写成 `timeline_fact`。之后新增测试并修复：

- 包含 `你记得/还记得/remember/recall` 的检索型问题不抽取事实。
- 带问号且包含 `什么/吗/是否/是不是` 的查询句不抽取事实。
- 本轮生成的污染记录已标记为 `status=inactive`，不再参与检索。

## 验证

自动测试：

```powershell
python -m unittest discover -s test
```

结果：

```text
Ran 40 tests in 1.329s
OK
```

新增覆盖：

- 带明确日期的计划会写入 `timeline_facts.jsonl`。
- 普通“今天随便聊聊”不会写入事实。
- 日期查询句不会写入事实。
- 用户消息入库后、后台 finalize 前即可检索 timeline fact。
- 日期 query 中，同源 timeline fact 会压过并去重 event summary。

## 当前局限

- 目前事实抽取是保守规则，不是 LLM 结构化抽取；优点是快、稳定、不阻塞，缺点是表达覆盖有限。
- `object` 仍可能保留“真实时间线最终测试：”这类测试前缀，后续可做更细的谓词/宾语清洗。
- 当前只实现 `has_time_bound_plan`，后续可以扩展到偏好变更、身份事实、目标、地点、无效化事实。
- 如果用户修改计划，例如“7月5日不用整理周报了”，还需要后续实现 invalidation/update 逻辑。

## 结论

本次实现解决了“日期相关事实只靠 summary 不够精确”的问题。实际运行中，第二轮日期追问已经优先命中 `timeline_fact`，并且 debug report 能展示完整分数组件和时间规则。事实层保持为 summary 的补充层，通过 source id 去重避免 prompt 重复，同时不影响后台 LLM summary 的异步处理。
