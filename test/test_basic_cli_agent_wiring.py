from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.basic_cli_chat as cli


class FakeSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.session_id = "session-1"

    def settle_memory_for_shutdown(self) -> dict:
        return {"status": "done"}


class FakeRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.closed = False

    def close(self) -> None:
        self.closed = True


class BasicCliAgentWiringTest(unittest.TestCase):
    def test_chat_and_task_tools_are_separated(self) -> None:
        captured = {}

        def make_session(**kwargs):
            captured["session"] = FakeSession(**kwargs)
            return captured["session"]

        def make_runner(*args, **kwargs):
            captured["runner"] = FakeRunner(*args, **kwargs)
            return captured["runner"]

        persona = SimpleNamespace(
            system_prompt="persona",
            expression_tags=(),
        )
        identity = SimpleNamespace(agent_id="default")

        with (
            patch.object(cli, "_load_env_file"),
            patch.dict(
                cli.os.environ,
                {"Guga_DEBUG": "0", "Guga_MODEL_ID": "fake-model"},
                clear=False,
            ),
            patch.object(cli, "create_chat_model", return_value=object()),
            patch.object(cli.PersonaManager, "load", return_value=persona),
            patch.object(cli, "identity_from_persona", return_value=identity),
            patch.object(cli, "MemoryManager", return_value=object()),
            patch.object(cli, "ChatSession", side_effect=make_session),
            patch.object(cli, "AgentTaskRunner", side_effect=make_runner, create=True),
            patch("builtins.input", return_value="/exit"),
        ):
            cli.main()

        chat_tools = captured["session"].kwargs["tool_registry"]
        task_tools = captured["runner"].args[1]
        self.assertEqual({"guga_parse_time"}, chat_tools.names())
        self.assertIn("guga_read_file", task_tools.names())
        self.assertIn("guga_run_command", task_tools.names())
        self.assertTrue(captured["runner"].closed)


if __name__ == "__main__":
    unittest.main()
