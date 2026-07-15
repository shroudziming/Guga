from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guga.persona.manager import PersonaManager
from guga.persona.skill_loader import load_persona_skill
from guga.utils.paths import personas_dir


class PersonaSkillLoaderTest(unittest.TestCase):
    def test_skill_backed_persona_loads_complete_body_without_frontmatter(self) -> None:
        source_skill = personas_dir() / "default" / "SKILL.md"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guga").mkdir()
            (root / "guga" / "SKILL.md").write_text(
                source_skill.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "guga.json").write_text(
                json.dumps(
                    {
                        "name": "guga",
                        "agent_id": "default",
                        "description": "Guga",
                        "skill_path": "guga/SKILL.md",
                    }
                ),
                encoding="utf-8",
            )
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
            outside.write_text(
                "---\nname: outside\ndescription: Use when testing\n---\nbody",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes personas directory"):
                load_persona_skill(outside, root)

    def test_skill_requires_nonempty_body_and_unique_allowed_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.md"
            empty.write_text(
                "---\nname: empty\ndescription: Use when testing\n---\n",
                encoding="utf-8",
            )
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


if __name__ == "__main__":
    unittest.main()
