from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guga.workspace import WorkspaceContext, WorkspaceError, workspace_context_from_env


class WorkspaceContextTest(unittest.TestCase):
    def test_inspect_set_reinspect_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default = root / "default"
            other = root / "other"
            default.mkdir()
            context = WorkspaceContext(default, allow_create=True)

            with self.assertRaisesRegex(WorkspaceError, "inspect"):
                context.require_confirmed()

            inspected = context.inspect()
            self.assertEqual(default.resolve(), Path(inspected["current_root"]))
            self.assertTrue(context.confirmed)

            changed = context.set(str(other), create_if_missing=True)
            self.assertEqual(default.resolve(), Path(changed["previous_root"]))
            self.assertEqual(other.resolve(), Path(changed["current_root"]))
            self.assertFalse(context.confirmed)
            self.assertTrue(other.is_dir())

            context.inspect()
            reset = context.reset()
            self.assertEqual(default.resolve(), Path(reset["current_root"]))
            self.assertEqual(default.resolve(), context.current_root)
            self.assertFalse(context.confirmed)

    def test_failed_switch_preserves_root_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default = Path(temp_dir) / "default"
            default.mkdir()
            context = WorkspaceContext(default, allow_create=False)
            context.inspect()

            with self.assertRaisesRegex(WorkspaceError, "does not exist"):
                context.set(
                    str(Path(temp_dir) / "missing"),
                    create_if_missing=True,
                )

            self.assertEqual(default.resolve(), context.current_root)
            self.assertTrue(context.confirmed)

    def test_relative_switch_and_environment_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            default = Path(temp_dir) / "default"
            child = default / "child"
            child.mkdir(parents=True)
            context = workspace_context_from_env(
                Path(temp_dir) / "fallback",
                {
                    "Guga_CLI_DEFAULT_WORKSPACE_PATH": str(default),
                    "Guga_CLI_ALLOW_CREATE_WORKSPACE": "1",
                },
            )

            context.inspect()
            context.set("child")

            self.assertEqual(child.resolve(), context.current_root)
            self.assertTrue(context.allow_create)


if __name__ == "__main__":
    unittest.main()
