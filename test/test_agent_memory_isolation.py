from __future__ import annotations

from contextlib import ExitStack
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guga.chat.session import ChatSession
from guga.persona.manager import PersonaManager
from guga.types import GenerationConfig
from guga.utils.debug_reporter import FileDebugSink
from guga.utils.paths import debug_reports_dir, personas_dir

from guga.memory.agent_identity import AgentIdentity, agent_memory_root
from guga.memory.manager import MemoryManager


class _ReplyOnlyModel:
    def generate_reply(self, messages, gen):
        _ = messages, gen
        return "ok"


class AgentMemoryIsolationTest(unittest.TestCase):
    def test_persona_configs_explicitly_declare_memory_identity(self) -> None:
        for persona_name in ("default", "gentle", "rational"):
            with self.subTest(persona=persona_name):
                payload = json.loads((personas_dir() / f"{persona_name}.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["agent_id"], persona_name)
                if persona_name == "default":
                    self.assertTrue(str(payload["skill_path"]).strip())
                else:
                    self.assertTrue(str(payload["reflection_context"]).strip())

    def test_default_persona_config_uses_repository_skill(self) -> None:
        payload = json.loads((personas_dir() / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["skill_path"], "default/SKILL.md")
        persona = PersonaManager(personas_dir()).load("default")
        self.assertEqual(persona.system_prompt, persona.reflection_context)
        self.assertIn("## Reflection 写作协议", persona.system_prompt)

    def test_agent_identity_rejects_unsafe_agent_ids(self) -> None:
        for bad_agent_id in ("", "../escape", "two words", "semi;colon", "slash/name"):
            with self.subTest(agent_id=bad_agent_id):
                with self.assertRaises(ValueError):
                    AgentIdentity(
                        agent_id=bad_agent_id,
                        reflection_context="ctx",
                        persona_source="config/personas/default.json",
                        persona_fingerprint="fp",
                    )

    def test_persona_roots_are_disjoint_for_sessions_rag_and_debug_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_base = Path(tmp_dir) / "data" / "memory"
            session_dirs: set[Path] = set()
            rag_dirs: set[Path] = set()
            debug_dirs: set[Path] = set()

            with self._patch_memory_roots(memory_base):
                persona_manager = PersonaManager(personas_dir())
                for persona_name in ("default", "gentle", "rational"):
                    persona = persona_manager.load(persona_name)
                    self.assertTrue(persona.agent_id)
                    self.assertTrue(persona.reflection_context)
                    if persona_name == "default":
                        self.assertEqual(persona.reflection_context, persona.system_prompt)
                    else:
                        self.assertNotEqual(persona.reflection_context, persona.system_prompt)

                    identity = AgentIdentity(
                        agent_id=persona.agent_id,
                        reflection_context=persona.reflection_context,
                        persona_source=persona.source_path,
                        persona_fingerprint=persona.persona_fingerprint,
                    )
                    manager = MemoryManager(agent_identity=identity, model=_ReplyOnlyModel())
                    manager.record_user_message("sess_same", f"hello from {persona_name}")
                    sink = FileDebugSink(debug_reports_dir(identity.agent_id))
                    sink("[DEBUG][ChatSession][sess_same] test")

                    session_dirs.add(manager.session_store.session_dir)
                    self.assertIsNotNone(manager.rag_pipeline)
                    rag_dirs.add(manager.rag_pipeline.index_dir)
                    debug_root = debug_reports_dir(identity.agent_id)
                    debug_dirs.add(debug_root)

                    expected_root = agent_memory_root(identity.agent_id)
                    self.assertEqual(manager.memory_root, expected_root)
                    self.assertTrue((expected_root / "sessions" / "sess_same.jsonl").exists())
                    self.assertTrue(any(debug_root.glob("*.log")))

                    manifest = json.loads((expected_root / "agent_manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(set(manifest), {"schema_version", "agent_id", "created_at"})
                    self.assertEqual(manifest["schema_version"], 2)
                    self.assertEqual(manifest["agent_id"], identity.agent_id)

                    revisions = [
                        json.loads(line)
                        for line in (expected_root / "persona_revisions.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
                    self.assertEqual([row["fingerprint"] for row in revisions], [identity.persona_fingerprint])

            self.assertEqual(len(session_dirs), 3)
            self.assertEqual(len(rag_dirs), 3)
            self.assertEqual(len(debug_dirs), 3)

    def test_default_chat_session_uses_agents_default_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_base = Path(tmp_dir) / "data" / "memory"
            with self._patch_memory_roots(memory_base):
                session = ChatSession(
                    model=_ReplyOnlyModel(),
                    system_prompt="system",
                    generation=GenerationConfig(),
                )

            self.assertEqual(session.memory_manager.memory_root, memory_base / "agents" / "default")
            self.assertNotEqual(session.memory_manager.memory_root, memory_base)

    def test_default_manager_does_not_read_legacy_session_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_base = Path(tmp_dir) / "data" / "memory"
            legacy_session_dir = memory_base / "sessions"
            legacy_session_dir.mkdir(parents=True, exist_ok=True)
            legacy_session_file = legacy_session_dir / "sess_legacy.jsonl"
            legacy_session_file.write_text(
                json.dumps(
                    {
                        "id": "msg_legacy",
                        "session_id": "sess_legacy",
                        "role": "user",
                        "content": "旧根目录里的消息不该被读到",
                        "created_at": "2026-07-10T09:00:00+08:00",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self._patch_memory_roots(memory_base):
                manager = MemoryManager(
                    agent_identity=AgentIdentity(
                        agent_id="default",
                        reflection_context="default persona",
                        persona_source="config/personas/default.json",
                        persona_fingerprint="default-fp",
                    ),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )
                context = manager.prepare_context("刚才我说了什么", session_id="sess_legacy")

            self.assertEqual(context.hits, [])
            self.assertEqual(context.archival_memories, [])
            self.assertFalse((memory_base / "agents" / "default" / "sessions" / "sess_legacy.jsonl").exists())

    def test_schema_one_manifest_migrates_without_moving_existing_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_base = Path(tmp_dir) / "data" / "memory"
            root = memory_base / "agents" / "default"
            (root / "sessions").mkdir(parents=True)
            (root / "sessions" / "existing.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "agent_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "agent_id": "default",
                        "persona_source": "config/personas/default.json",
                        "persona_fingerprint": "inline-v1",
                        "created_at": "2026-07-10T10:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )

            with self._patch_memory_roots(memory_base):
                manager = MemoryManager(
                    agent_identity=AgentIdentity(
                        "default", "skill", "config/personas/default/SKILL.md", "skill-v2"
                    ),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )

            manifest = json.loads((manager.memory_root / "agent_manifest.json").read_text(encoding="utf-8"))
            revisions = [
                json.loads(line)
                for line in (manager.memory_root / "persona_revisions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(set(manifest), {"schema_version", "agent_id", "created_at"})
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue((manager.memory_root / "sessions" / "existing.jsonl").exists())
            self.assertEqual([row["fingerprint"] for row in revisions], ["inline-v1", "skill-v2"])

    def test_new_skill_revision_appends_once_and_reuses_same_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_base = Path(tmp_dir) / "data" / "memory"
            with self._patch_memory_roots(memory_base):
                first = MemoryManager(
                    agent_identity=AgentIdentity("default", "skill", "skill.md", "v1"),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )
                second = MemoryManager(
                    agent_identity=AgentIdentity("default", "skill", "skill.md", "v2"),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )
                MemoryManager(
                    agent_identity=AgentIdentity("default", "skill", "skill.md", "v2"),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )

            rows = [
                json.loads(line)
                for line in (first.memory_root / "persona_revisions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(first.memory_root, second.memory_root)
            self.assertEqual([row["fingerprint"] for row in rows], ["v1", "v2"])

    def test_different_agent_id_cannot_claim_existing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "shared-agent-root"
            MemoryManager(
                memory_root=root,
                agent_identity=AgentIdentity("default", "default", "default.md", "v1"),
                model=_ReplyOnlyModel(),
                enable_semantic=False,
            )

            with self.assertRaisesRegex(ValueError, "agent manifest mismatch for agent_id"):
                MemoryManager(
                    memory_root=root,
                    agent_identity=AgentIdentity("other", "other", "other.md", "v1"),
                    model=_ReplyOnlyModel(),
                    enable_semantic=False,
                )

    def _patch_memory_roots(self, memory_base: Path):
        stack = ExitStack()
        stack.enter_context(patch("guga.memory.manager.memory_data_dir", lambda: memory_base))
        stack.enter_context(patch("guga.memory.agent_identity.memory_data_dir", lambda: memory_base))
        return stack


if __name__ == "__main__":
    unittest.main()
