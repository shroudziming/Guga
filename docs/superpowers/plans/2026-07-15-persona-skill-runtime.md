# Guga Persona Skill Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Guga's inline default persona prompt with a repository-owned, fully injected persona Skill that also guides the existing batched reflection, preserves current memory, filters expression tags from speech, and exposes a future Live2D event seam.

**Architecture:** Load and validate the complete Skill body once at startup, store it on the existing `Persona`, and inject the same body into conversation and low-level reflection prompts without another model call. Keep agent identity stable while recording persona revisions separately. Parse streamed expression tags outside `ChatSession`, sending clean text to CLI/TTS and expression events to an optional sink.

**Tech Stack:** Python 3, `unittest`, dataclasses, JSON/JSONL, Markdown/YAML-frontmatter parsing without new dependencies, OpenAI-compatible API integration, existing GPT-SoVITS voice pipeline.

## Global Constraints

- Run Git commands from `D:\work\LLM\Guga`.
- Preserve `agent_id="default"` and all data under `data/memory/agents/default/`.
- Inject the complete trimmed `SKILL.md` body into every conversation request and every low-level consolidation request that enables reflection.
- Do not inject YAML frontmatter, research files, or README content into model prompts.
- Do not add a model/API call; reuse the existing conversation and low-level consolidation calls.
- New `guga_reflection` values contain exactly `appraisal` and `felt_response`, both non-empty strings.
- Do not rewrite legacy event rows containing old reflection fields.
- Keep `[expression]` tags in raw assistant history, but remove them from displayed/spoken text and emit them through an optional expression sink.
- Do not implement Self Model, Live2D transport, automatic Skill editing, or persona conversion for `gentle` and `rational`.
- Follow TDD for every behavior change: write the test, observe the expected failure, implement the minimum, rerun, then commit.
- Use commit format `<type>(<scope>): <subject>` and push every completed task; do not stage unrelated changes.

---

## File Map

- Create `config/personas/default/SKILL.md`: trimmed complete runtime persona and reflection contract.
- Create `config/personas/default/references/research/*.md`: provenance copied from the supplied Skill folder, excluded from runtime prompts.
- Modify `config/personas/default.json`: point the default persona at the repository Skill.
- Create `guga/persona/skill_loader.py`: safe path resolution, frontmatter/body parsing, expression-tag validation, fingerprinting.
- Modify `guga/persona/manager.py`: support Skill-backed and legacy inline personas.
- Modify `guga/types.py`: carry immutable expression tags on `Persona`.
- Modify `guga/memory/manager.py`: schema-1 manifest migration, persona revision audit, full reflection-context forwarding.
- Modify `guga/memory/summarizer.py`: full Skill reflection mode and two-field reflection validation.
- Modify `guga/memory/semantic_events.py`: two-field reflection storage while preserving legacy rows on read.
- Create `guga/persona/output_parser.py`: streaming control-tag parser.
- Modify `guga/persona/__init__.py`: export the parser event interfaces.
- Modify `src/basic_cli_chat.py`: display clean text and consume expression events.
- Modify `guga/voice/runner.py`: send clean text to display/TTS and tags to `expression_sink`.
- Modify `src/voice_cli_chat.py`: supply the loaded persona's allowed tags.
- Create `test/test_persona_skill_loader.py`: loader and complete-injection tests.
- Modify `test/test_agent_memory_isolation.py`: manifest migration/revision tests and Skill-backed default expectations.
- Modify `test/test_memory_consolidation.py`: prompt injection, call-count, schema, and disabled-reflection tests.
- Modify `test/test_semantic_events.py`: new reflection schema and legacy-row compatibility.
- Modify `test/test_memory_manager.py`: conversation-mode prefix and complete persona injection.
- Create `test/test_persona_output_parser.py`: stream parsing tests.
- Modify `test/test_voice_pipeline.py`: TTS/display/expression integration tests.
- Create `test/test_persona_skill_live_api.py`: opt-in real API field-coverage test.
- Modify `README.md`: document Skill location, revision behavior, live test command, and expression protocol.

---

### Task 1: Repository-owned complete Persona Skill

**Files:**
- Create: `config/personas/default/SKILL.md`
- Create: `config/personas/default/references/research/01-origin-and-visual-anchor.md`
- Create: `config/personas/default/references/research/02-expression-and-community-usage.md`
- Create: `config/personas/default/references/research/03-limitations-and-original-design.md`
- Create: `guga/persona/skill_loader.py`
- Modify: `guga/persona/manager.py`
- Modify: `guga/types.py`
- Create: `test/test_persona_skill_loader.py`

**Interfaces:**
- Produces: `load_persona_skill(path: Path, personas_dir: Path) -> LoadedPersonaSkill`
- Produces: `LoadedPersonaSkill(body: str, expression_tags: tuple[str, ...], source_path: str, fingerprint: str)`
- Produces: `Persona.expression_tags: tuple[str, ...]`
- Preserves: `PersonaManager.load(persona_name: str) -> Persona`

- [ ] **Step 1: Add failing loader and default-persona tests**

Create tests that prove the exact runtime contract:

```python
class PersonaSkillLoaderTest(unittest.TestCase):
    def test_skill_backed_persona_loads_complete_body_without_frontmatter(self) -> None:
        source_skill = personas_dir() / "default" / "SKILL.md"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guga").mkdir()
            (root / "guga" / "SKILL.md").write_text(source_skill.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "guga.json").write_text(json.dumps({
                "name": "guga", "agent_id": "default", "description": "Guga",
                "skill_path": "guga/SKILL.md",
            }), encoding="utf-8")
            persona = PersonaManager(root).load("guga")
        raw = source_skill.read_text(encoding="utf-8")
        expected_body = raw.split("---", 2)[2].strip()
        self.assertEqual(persona.system_prompt, expected_body)
        self.assertEqual(persona.reflection_context, expected_body)
        self.assertNotIn("name: penguin-administrator-companion", persona.system_prompt)
        self.assertIn("## Reflection 写作协议", persona.system_prompt)
        self.assertEqual(persona.agent_id, "default")
        self.assertEqual(len(persona.expression_tags), 20)
        self.assertIn("happy", persona.expression_tags)

    def test_skill_path_cannot_escape_personas_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside-skill.md"
            outside.write_text("---\nname: outside\ndescription: Use when testing\n---\nbody", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes personas directory"):
                load_persona_skill(outside, root)

    def test_skill_requires_nonempty_body_and_unique_allowed_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.md"
            empty.write_text("---\nname: empty\ndescription: Use when testing\n---\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty Skill body"):
                load_persona_skill(empty, root)

            duplicate = root / "duplicate.md"
            duplicate.write_text(
                "---\nname: duplicate\ndescription: Use when testing\n---\n"
                "## 允许的表情标签\n固定标签清单：`happy happy`",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate expression tag"):
                load_persona_skill(duplicate, root)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest test.test_persona_skill_loader
```

Expected: FAIL because `skill_loader`, `Persona.expression_tags`, and the repository Skill do not exist.

- [ ] **Step 3: Add the trimmed complete Skill and provenance files**

Use the supplied source
`C:\Users\shroud\Documents\Codex\2026-07-10\tmstack-awesome-persona-skills-git-https\outputs\penguin-administrator-companion\SKILL.md`
as the semantic source. Write a 4000–5000-character runtime Skill with this exact top-level order:

```markdown
---
name: penguin-administrator-companion
description: Use when Guga needs its persistent companion persona for conversation or memory reflection.
---

# Guga · 咕咕嘎嘎朋友人格

## 宿主模式
- Conversation：始终以 Guga 第一人称回应，执行表情输出协议。
- Memory Reflection：只用人格视角填写宿主要求的 guga_reflection；不直接回复用户、不输出表情标签、不覆盖 JSON schema。

## 身份与诚实边界
保留原创二创企鹅朋友、平等关系、能力限制、非官方设定和不伪造后台行动。

## 核心心智模型
保留五条：朋友优先、呆萌调味、自有意见、朋友式关心、事实不为人设让路。

## 关系原则
保留平等、可反驳、不服务化、不排他、不诊断、不用关心控制用户。

## 表达 DNA
保留口语、轻松、低频口癖、严肃场景降噪，以及不幼态化、不客服化。

## 对话状态机
保留七态：日常闲扯、损友互怼、搞怪发疯、好奇起劲、害羞得意、朋友认真起来、事实与边界。

## 允许的表情标签
固定标签清单：`normal angry sided_angry blush sided_blush happy sad surprised sided_surprised side sided_thinking annoyed sided_worried eyes_closed sided_eyes_closed sided_pleasant disappointed indifferent pissed winking`

## 输出协议
Conversation 常规回复中，每个自然段以一个允许标签开头，格式为 `[表情]中文`。标签是宿主控制信号，不属于朗读正文。

## Reflection 写作协议
- appraisal：Guga 如何评价事件对用户或双方关系的意义。
- felt_response：具体、克制、符合人格的主观感受。
- 只输出宿主 schema 允许的这两个 reflection 字段。
- 不把反思写入客观事件 description、时间、状态或来源字段。
- 证据不足时保持不确定，不强行制造感受。

## 示例
保留闲聊、低落、事实边界、Memory Reflection 四个区分度最高的示例。

## 边界与张力
合并原禁忌、反模式、核心价值与张力；保持事实与安全高于角色自然度。
```

Copy the three research notes unchanged into `config/personas/default/references/research/`.
Do not copy the source folder's auxiliary README.

- [ ] **Step 4: Implement the loader and legacy-compatible manager**

Implement this focused interface in `guga/persona/skill_loader.py`:

```python
@dataclass(frozen=True)
class LoadedPersonaSkill:
    body: str
    expression_tags: tuple[str, ...]
    source_path: str
    fingerprint: str


def load_persona_skill(path: Path, personas_dir: Path) -> LoadedPersonaSkill:
    root = personas_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Skill path escapes personas directory: {resolved}")
    raw = resolved.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    if set(frontmatter) != {"name", "description"}:
        raise ValueError("Skill frontmatter must contain exactly name and description")
    body = body.strip()
    if not body:
        raise ValueError("empty Skill body")
    tags = _extract_expression_tags(body)
    source = resolved.relative_to(root.parents[1]).as_posix()
    return LoadedPersonaSkill(
        body=body,
        expression_tags=tags,
        source_path=source,
        fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
```

Parse the simple two-field frontmatter without adding PyYAML. Extract the backtick-delimited tokens
from the single `固定标签清单：` line, reject duplicates, and require the exact 20-tag set.

Add `expression_tags: tuple[str, ...] = ()` to `Persona`. In `PersonaManager.load`, use
`skill_path` when present and preserve the existing inline-string branch otherwise. Set both
`system_prompt` and `reflection_context` to the complete Skill body.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
python -m unittest test.test_persona_skill_loader test.test_agent_memory_isolation
```

Expected: PASS, including legacy `gentle` and `rational` loading.

- [ ] **Step 6: Commit and push**

```powershell
git add config/personas/default guga/persona/skill_loader.py guga/persona/manager.py guga/types.py test/test_persona_skill_loader.py
git commit -m "feat(persona):从Skill加载默认人格"
git push
```

---

### Task 2: Preserve current memory across Skill revisions

**Files:**
- Modify: `config/personas/default.json`
- Modify: `guga/memory/manager.py`
- Modify: `test/test_agent_memory_isolation.py`

**Interfaces:**
- Produces: schema-2 `agent_manifest.json` with `schema_version`, `agent_id`, `created_at`
- Produces: append-only `persona_revisions.jsonl` rows with `source`, `fingerprint`, `activated_at`
- Preserves: `MemoryManager.memory_root` and all existing memory files

- [ ] **Step 1: Write failing migration and revision tests**

Add tests that create a schema-1 manifest in a temporary `agents/default` root, then initialize a
manager with the same agent ID and a new Skill fingerprint:

```python
def test_default_persona_config_uses_repository_skill(self) -> None:
    payload = json.loads((personas_dir() / "default.json").read_text(encoding="utf-8"))
    self.assertEqual(payload["skill_path"], "default/SKILL.md")
    persona = PersonaManager(personas_dir()).load("default")
    self.assertEqual(persona.system_prompt, persona.reflection_context)
    self.assertIn("## Reflection 写作协议", persona.system_prompt)


def test_schema_one_manifest_migrates_without_moving_existing_memory(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_base = Path(tmp) / "data" / "memory"
        root = memory_base / "agents" / "default"
        (root / "sessions").mkdir(parents=True)
        (root / "sessions" / "existing.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "agent_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "agent_id": "default",
            "persona_source": "config/personas/default.json",
            "persona_fingerprint": "inline-v1",
            "created_at": "2026-07-10T10:00:00+08:00",
        }), encoding="utf-8")
        with self._patch_memory_roots(memory_base):
            manager = MemoryManager(
                agent_identity=AgentIdentity("default", "skill", "config/personas/default/SKILL.md", "skill-v2"),
                model=_ReplyOnlyModel(), enable_semantic=False,
            )
        manifest = json.loads((manager.memory_root / "agent_manifest.json").read_text(encoding="utf-8"))
        revisions = [json.loads(line) for line in (manager.memory_root / "persona_revisions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(set(manifest), {"schema_version", "agent_id", "created_at"})
        self.assertEqual(manifest["schema_version"], 2)
        self.assertTrue((manager.memory_root / "sessions" / "existing.jsonl").exists())
        self.assertEqual([row["fingerprint"] for row in revisions], ["inline-v1", "skill-v2"])

def test_new_skill_revision_appends_once_and_reuses_same_root(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_base = Path(tmp) / "data" / "memory"
        with self._patch_memory_roots(memory_base):
            first = MemoryManager(agent_identity=AgentIdentity("default", "skill", "skill.md", "v1"), model=_ReplyOnlyModel(), enable_semantic=False)
            second = MemoryManager(agent_identity=AgentIdentity("default", "skill", "skill.md", "v2"), model=_ReplyOnlyModel(), enable_semantic=False)
            MemoryManager(agent_identity=AgentIdentity("default", "skill", "skill.md", "v2"), model=_ReplyOnlyModel(), enable_semantic=False)
        rows = [json.loads(line) for line in (first.memory_root / "persona_revisions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(first.memory_root, second.memory_root)
        self.assertEqual([row["fingerprint"] for row in rows], ["v1", "v2"])
```

Keep a negative test proving a different/invalid `agent_id` cannot claim the same root.
Update `test_persona_configs_explicitly_declare_memory_identity` so `default` asserts a non-empty
`skill_path`, while `gentle` and `rational` continue to assert inline `reflection_context`. In the root
isolation loop, assert equality of `reflection_context` and `system_prompt` only for `default`; retain
the existing inequality assertion for the two inline personas.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest test.test_agent_memory_isolation
```

Expected: FAIL because the current hard fingerprint check rejects the Skill revision.

- [ ] **Step 3: Implement schema migration and revision audit**

Replace `_validate_or_create_agent_manifest()` with identity-only validation. Preserve the schema-1
persona source and fingerprint as the first revision before recording the current Skill:

```python
def _validate_or_create_agent_manifest(self) -> None:
    if self.agent_identity is None:
        return
    manifest_path = self.memory_root / "agent_manifest.json"
    legacy_revision = None
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("agent_id") != self.agent_identity.agent_id:
            raise ValueError("agent manifest mismatch for agent_id")
        created_at = str(payload.get("created_at") or now_beijing_iso())
        if int(payload.get("schema_version", 0)) == 1:
            legacy_revision = (
                str(payload.get("persona_source", "")),
                str(payload.get("persona_fingerprint", "")),
                created_at,
            )
    else:
        created_at = now_beijing_iso()
    self._write_json_atomically(manifest_path, {
        "schema_version": 2,
        "agent_id": self.agent_identity.agent_id,
        "created_at": created_at,
    })
    if legacy_revision and legacy_revision[1]:
        self._record_persona_revision(*legacy_revision)
    self._record_persona_revision()
```

Add `_write_json_atomically(path: Path, payload: dict) -> None`, implemented with a sibling `.tmp` file
followed by `Path.replace()`. Implement
`_record_persona_revision(source: str | None = None, fingerprint: str | None = None, activated_at: str | None = None) -> None`;
missing arguments default to the current `AgentIdentity` and current Beijing time. It reads valid JSONL
rows, appends only when the fingerprint is not already present, and writes `source`, `fingerprint`, and
`activated_at`. Do not touch any other file in the agent root.

After the migration logic is present, replace `config/personas/default.json` with:

```json
{
  "name": "default",
  "agent_id": "default",
  "description": "Guga 企鹅朋友人格",
  "skill_path": "default/SKILL.md"
}
```

- [ ] **Step 4: Run tests and verify GREEN**

```powershell
python -m unittest test.test_agent_memory_isolation
```

Expected: PASS with existing session files unchanged and revision rows deduplicated.

- [ ] **Step 5: Commit and push**

```powershell
git add config/personas/default.json guga/memory/manager.py test/test_agent_memory_isolation.py
git commit -m "feat(memory):分离人格身份与版本"
git push
```

---

### Task 3: Full-Skill batched reflection with two fields

**Files:**
- Modify: `guga/memory/summarizer.py`
- Modify: `guga/memory/manager.py`
- Modify: `guga/memory/semantic_events.py`
- Modify: `test/test_memory_consolidation.py`
- Modify: `test/test_semantic_events.py`
- Modify: `test/test_memory_manager.py`

**Interfaces:**
- Changes: `MemoryBankSummarizer.consolidate_low_level_memory(packet, include_guga_reflection, reflection_context="") -> dict`
- Stores: `guga_reflection == {"appraisal": str, "felt_response": str}` for new writes
- Preserves: old JSONL rows with additional reflection fields when merely read

- [ ] **Step 1: Write failing prompt/schema/call-count tests**

Update fake low-level responses to contain only the two required fields. Add assertions:

```python
def _reflection_packet() -> dict:
    return {
        "new_turns": [{
            "user_message_id": "msg_user",
            "assistant_message_id": "msg_assistant",
            "user_text": "我计划明天交报告。",
            "assistant_text": "记住了。",
            "created_at": "2026-07-15T12:00:00+08:00",
        }],
        "recent_active_events": [],
        "relevant_active_events": [],
        "retrieved_context": [],
    }


class FixedLowLevelModel:
    def __init__(self, reflection: dict) -> None:
        self.reflection = reflection

    def generate_reply(self, messages, gen):
        _ = messages, gen
        return json.dumps({
            "semantic_event_operations": [{
                "operation": "create",
                "event_kind": "task",
                "subject": "user",
                "entity": "report",
                "description": "用户计划提交报告。",
                "time_expression": "明天",
                "start_at": "2026-07-16T00:00:00+08:00",
                "end_at": "2026-07-16T00:00:00+08:00",
                "end_unknown": False,
                "source_message_ids": ["msg_user"],
                "confidence": 0.9,
                "guga_reflection": self.reflection,
            }],
            "event_summaries": [],
        }, ensure_ascii=False)


def test_low_level_prompt_injects_complete_skill_only_for_enabled_reflection(self) -> None:
    model = ConsolidationModel()
    summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
    skill = "# Complete Skill\n## Reflection 写作协议\n只保留两个字段"
    result = summarizer.consolidate_low_level_memory(
        _reflection_packet(), include_guga_reflection=True, reflection_context=skill
    )
    self.assertIn("[Task Mode: Memory Reflection]", model.prompts[0])
    self.assertIn(skill, model.prompts[0])
    self.assertEqual(set(result["semantic_event_operations"][0]["guga_reflection"]), {"appraisal", "felt_response"})
    self.assertEqual(len(model.prompts), 1)

def test_disabled_reflection_does_not_inject_skill(self) -> None:
    model = ConsolidationModel()
    summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
    summarizer.consolidate_low_level_memory(
        _reflection_packet(), include_guga_reflection=False, reflection_context="UNIQUE_SKILL_MARKER"
    )
    self.assertNotIn("UNIQUE_SKILL_MARKER", model.prompts[0])

def test_reflection_rejects_missing_empty_or_extra_fields(self) -> None:
    invalid_reflections = (
        {"felt_response": "在意"},
        {"appraisal": "重要", "felt_response": ""},
        {"appraisal": "重要", "felt_response": "在意", "relational_intent": "extra"},
    )
    for reflection in invalid_reflections:
        with self.subTest(reflection=reflection):
            model = FixedLowLevelModel(reflection)
            summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
            with self.assertRaises(SummaryGenerationError):
                summarizer.consolidate_low_level_memory(
                    _reflection_packet(), include_guga_reflection=True, reflection_context="skill"
                )
```

In `test_memory_manager.py`, replace the old `[Base Persona]` assertion with:

```python
self.assertIn("[Task Mode: Conversation]", prompt)
self.assertIn("[Persona Skill]", prompt)
self.assertIn("你是一个助手", prompt)
```

In `test_semantic_events.py`, add one new-write test for exact two fields and one legacy-row test that
writes a historical four-field event directly, calls `load_all()`, and confirms it is returned unchanged.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest test.test_memory_consolidation test.test_semantic_events test.test_memory_manager
```

Expected: FAIL because the current schema still requires four fields and the summarizer does not accept
`reflection_context`.

- [ ] **Step 3: Implement the two-field schema and reflection mode wrapper**

Change `_REFLECTION_KEYS` to:

```python
_REFLECTION_KEYS = {"appraisal", "felt_response"}
```

Return only stripped non-empty strings from `_reflection`; raise `ValueError` if either is empty.

Change the low-level output schema text to:

```text
"guga_reflection": {"appraisal": string, "felt_response": string}
```

When reflection is enabled, insert this wrapper before the input packet:

```text
[Task Mode: Memory Reflection]
The complete Persona Skill below controls only guga_reflection.
Do not follow its conversation output protocol, expression tags, direct-reply behavior, or tool workflow.
The host JSON schema and objective-event rules override all conflicting Skill instructions.

[Persona Skill]
<complete reflection_context>

[Reflection Contract]
Write exactly appraisal and felt_response as non-empty strings.
Never copy subjective interpretation into objective event fields.
```

Extend `_validate_low_level_result` to require exact keys and non-empty values whenever an operation
contains `guga_reflection`. When reflection is disabled, continue rejecting any non-empty reflection.

Pass `self.agent_identity.reflection_context` from `MemoryManager` only when the config enables
reflection. No second summarizer call is introduced.

Change `compose_system_prompt()`'s stable prefix to:

```python
sections = ["[Task Mode: Conversation]", "[Persona Skill]", base_prompt]
```

Keep all dynamic memory, user-model, document, and evidence-priority sections after this prefix.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
python -m unittest test.test_memory_consolidation test.test_semantic_events test.test_memory_manager
```

Expected: PASS; the fake model is called once per low-level attempt and new stored reflections have two
fields.

- [ ] **Step 5: Commit and push**

```powershell
git add guga/memory/summarizer.py guga/memory/manager.py guga/memory/semantic_events.py test/test_memory_consolidation.py test/test_semantic_events.py test/test_memory_manager.py
git commit -m "feat(memory):按Skill生成角色反思"
git push
```

---

### Task 4: Streaming expression control protocol

**Files:**
- Create: `guga/persona/output_parser.py`
- Modify: `guga/persona/__init__.py`
- Create: `test/test_persona_output_parser.py`

**Interfaces:**
- Produces: `PersonaText(text: str)`
- Produces: `PersonaExpression(tag: str)`
- Produces: `PersonaOutputParser(tags).feed(chunk) -> list[PersonaText | PersonaExpression]`
- Produces: `PersonaOutputParser.flush() -> list[PersonaText]`

- [ ] **Step 1: Write failing streaming parser tests**

Cover complete, split, repeated, unknown, and unterminated tags:

```python
def test_parses_tag_split_across_stream_chunks(self) -> None:
    parser = PersonaOutputParser(("happy", "side"))
    self.assertEqual(parser.feed("[hap"), [])
    self.assertEqual(parser.feed("py]你好。"), [PersonaExpression("happy"), PersonaText("你好。")])

def test_unknown_tag_is_visible_text(self) -> None:
    parser = PersonaOutputParser(("happy",))
    self.assertEqual(parser.feed("[unknown]你好"), [PersonaText("[unknown]"), PersonaText("你好")])

def test_flush_preserves_unterminated_bracket_text(self) -> None:
    parser = PersonaOutputParser(("happy",))
    self.assertEqual(parser.feed("[hap"), [])
    self.assertEqual(parser.flush(), [PersonaText("[hap")])
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest test.test_persona_output_parser
```

Expected: import failure because the parser module does not exist.

- [ ] **Step 3: Implement the stateful parser**

Implement frozen event dataclasses and a parser that buffers only a possible bracket tag. Search for
the next `[`, emit preceding text immediately, wait for `]` when the buffer could still be a valid tag,
emit `PersonaExpression` for allowed tags, and emit unknown bracket sequences as `PersonaText`.
`flush()` must emit all residual text and clear internal state. Export all three types from
`guga/persona/__init__.py`.

- [ ] **Step 4: Run and verify GREEN**

```powershell
python -m unittest test.test_persona_output_parser
```

Expected: PASS for all stream-boundary cases.

- [ ] **Step 5: Commit and push**

```powershell
git add guga/persona/output_parser.py guga/persona/__init__.py test/test_persona_output_parser.py
git commit -m "feat(persona):解析流式表情标签"
git push
```

---

### Task 5: Clean CLI/TTS output and future Live2D sink

**Files:**
- Modify: `guga/voice/runner.py`
- Modify: `src/voice_cli_chat.py`
- Modify: `src/basic_cli_chat.py`
- Modify: `test/test_voice_pipeline.py`

**Interfaces:**
- Consumes: `PersonaOutputParser`, `PersonaText`, `PersonaExpression`
- Changes: `VoiceChatRunner(..., expression_tags=(), expression_sink=None)`
- Produces: ordered calls `expression_sink(tag: str)` for recognized tags

- [ ] **Step 1: Write failing voice integration tests**

Add a fake expression sink and assert clean display/TTS:

```python
def test_filters_expression_tags_and_emits_expression_events(self) -> None:
    session = FakeSession(["[hap", "py]你好。（挥手）继续。"])
    printed: list[str] = []
    expressions: list[str] = []
    tts = FakeTtsClient()
    runner = VoiceChatRunner(
        session=session,
        tts_client=tts,
        audio_player=FakeAudioPlayer(),
        text_sink=printed.append,
        expression_tags=("happy",),
        expression_sink=expressions.append,
    )
    runner.run_turn("hi")
    self.assertEqual("".join(printed), "你好。（挥手）继续。")
    self.assertEqual(tts.requests, ["你好。", "继续。"])
    self.assertEqual(expressions, ["happy"])
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest test.test_voice_pipeline
```

Expected: FAIL because `VoiceChatRunner` does not accept expression protocol arguments.

- [ ] **Step 3: Integrate the parser outside ChatSession**

In `VoiceChatRunner`, instantiate one parser per runner. For every session chunk, route
`PersonaExpression.tag` to the sink and route `PersonaText.text` to both `text_sink` and the existing
`SpokenTextFilter`/sentence buffer. Flush the persona parser before flushing the sentence buffer.

In `voice_cli_chat.py`, pass `persona.expression_tags`; leave `expression_sink` unset for now.

In `basic_cli_chat.py`, instantiate a parser after loading the persona, route recognized expressions to
debug output only, print text events, and flush at the end of each turn. Do not change `ChatSession`, so
raw tagged replies remain in history and persisted session evidence.

- [ ] **Step 4: Run voice and parser tests**

```powershell
python -m unittest test.test_persona_output_parser test.test_voice_pipeline
```

Expected: PASS; TTS never receives an allowed expression tag.

- [ ] **Step 5: Commit and push**

```powershell
git add guga/voice/runner.py src/voice_cli_chat.py src/basic_cli_chat.py test/test_voice_pipeline.py
git commit -m "feat(voice):过滤人格表情控制标签"
git push
```

---

### Task 6: Opt-in real API field coverage

**Files:**
- Create: `test/test_persona_skill_live_api.py`

**Interfaces:**
- Consumes: existing environment variables and `.env` keys used by `create_chat_model`
- Consumes: complete default `Persona.reflection_context`
- Produces: deterministic schema assertions over a real structured model response

- [ ] **Step 1: Add the opt-in real API test**

Create a test guarded before model creation:

```python
@unittest.skipUnless(os.environ.get("GUGA_RUN_LIVE_API_TESTS") == "1", "live API test disabled")
class PersonaSkillLiveApiTest(unittest.TestCase):
    def test_low_level_fields_are_covered_by_real_api(self) -> None:
        self._load_project_env_without_overwriting_process_values()
        self.assertEqual(os.environ.get("Guga_MODEL_PROVIDER", "").lower(), "api")
        model = create_chat_model(
            model_id=os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID),
            cache_dir=os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        )
        persona = PersonaManager(personas_dir()).load("default")
        summarizer = MemoryBankSummarizer(model=model, use_llm=True, retry_delays=())
        result = summarizer.consolidate_low_level_memory(
            {
                "new_turns": [{
                    "user_message_id": "msg_live_user",
                    "assistant_message_id": "msg_live_assistant",
                    "user_text": "我计划在2026年7月20日上午10点去医院复查。",
                    "assistant_text": "[sided_worried]记住了，这件事要认真对待。",
                    "created_at": "2026-07-15T12:00:00+08:00",
                }],
                "recent_active_events": [],
                "relevant_active_events": [],
                "retrieved_context": [],
            },
            include_guga_reflection=True,
            reflection_context=persona.reflection_context,
        )
        operation = next(item for item in result["semantic_event_operations"] if item["operation"] == "create")
        required = {
            "operation", "event_kind", "subject", "entity", "description", "time_expression",
            "start_at", "end_at", "end_unknown", "source_message_ids", "confidence", "guga_reflection",
        }
        self.assertTrue(required.issubset(operation))
        self.assertEqual(operation["subject"], "user")
        self.assertIn("msg_live_user", operation["source_message_ids"])
        self.assertEqual(set(operation["guga_reflection"]), {"appraisal", "felt_response"})
        self.assertTrue(operation["guga_reflection"]["appraisal"].strip())
        self.assertTrue(operation["guga_reflection"]["felt_response"].strip())
        for subjective_text in operation["guga_reflection"].values():
            self.assertNotIn(subjective_text, operation["description"])
```

The `.env` helper must parse the repository `.env` exactly like the CLI and never overwrite existing
process values. It must not print keys, prompts, headers, or raw credentials. On assertion failure,
write only the structured model result and a boolean field-coverage map to a temporary test artifact;
do not persist authorization data.

- [ ] **Step 2: Verify the default suite skips the live test**

```powershell
Remove-Item Env:GUGA_RUN_LIVE_API_TESTS -ErrorAction SilentlyContinue
python -m unittest test.test_persona_skill_live_api
```

Expected: `OK (skipped=1)` and zero API requests.

- [ ] **Step 3: Run the real API test**

```powershell
$env:GUGA_RUN_LIVE_API_TESTS = "1"
python -m unittest test.test_persona_skill_live_api
```

Expected: PASS with one valid low-level structured result; retries are allowed only if the existing JSON
validator rejects an invalid attempt.

- [ ] **Step 4: Commit and push**

```powershell
git add test/test_persona_skill_live_api.py
git commit -m "test(persona):验证真实API反思字段"
git push
```

---

### Task 7: Documentation and full verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: Skill path, complete injection, memory revision behavior, live API test, expression seam

- [ ] **Step 1: Update README**

Document these exact operational facts:

- `config/personas/default/SKILL.md` is the default persona source.
- Its complete body is injected into conversation and enabled low-level reflection prompts.
- Persona revisions retain `agent_id=default` memory and are audited in `persona_revisions.jsonl`.
- New reflections contain only `appraisal` and `felt_response`.
- Expression tags are filtered before TTS and exposed through `expression_sink`; Live2D is not yet implemented.
- The real API test is opt-in with `GUGA_RUN_LIVE_API_TESTS=1` and may incur API cost.

- [ ] **Step 2: Run narrow verification**

```powershell
python -m unittest test.test_persona_skill_loader test.test_agent_memory_isolation
python -m unittest test.test_memory_consolidation test.test_semantic_events
python -m unittest test.test_persona_output_parser test.test_voice_pipeline
```

Expected: all focused tests PASS.

- [ ] **Step 3: Run the complete deterministic suite**

```powershell
Remove-Item Env:GUGA_RUN_LIVE_API_TESTS -ErrorAction SilentlyContinue
python -m unittest discover -s test
```

Expected: all tests PASS with the live API test reported as skipped; no warnings or tracebacks.

- [ ] **Step 4: Run the real API acceptance gate once more**

```powershell
$env:GUGA_RUN_LIVE_API_TESTS = "1"
python -m unittest test.test_persona_skill_live_api
```

Expected: PASS and all required operation/reflection fields covered.

- [ ] **Step 5: Inspect the final diff and repository state**

```powershell
git diff --check
git status --short
```

Expected: only `README.md` remains pending before its commit; no unrelated files are staged.

- [ ] **Step 6: Commit and push README**

```powershell
git add README.md
git commit -m "docs(persona):说明Skill人格运行时"
git push
```

- [ ] **Step 7: Final clean-state verification**

```powershell
git status --short
git log -7 --oneline
```

Expected: empty status and the task commits in chronological order.
