from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import src.basic_cli_chat as basic_cli
import src.voice_cli_chat as voice_cli


class CliShutdownTest(unittest.TestCase):
    def _fake_session(self) -> MagicMock:
        session = MagicMock()
        session.settle_memory_for_shutdown.return_value = {"status": "complete"}
        return session

    def _persona(self) -> MagicMock:
        return MagicMock(
            system_prompt="test",
            reflection_context="test",
            source_path="test",
            persona_fingerprint="test",
            expression_tags=(),
        )

    def _run_basic(self, read_side_effect) -> MagicMock:
        session = self._fake_session()
        with (
            patch.dict(os.environ, {"Guga_DEBUG": "0"}, clear=False),
            patch.object(basic_cli, "_load_env_file"),
            patch.object(basic_cli, "read_user_input", side_effect=read_side_effect),
            patch.object(basic_cli, "create_chat_model", return_value=MagicMock()),
            patch.object(basic_cli, "MemoryManager", return_value=MagicMock()),
            patch.object(basic_cli, "ChatSession", return_value=session),
            patch.object(basic_cli.PersonaManager, "load", return_value=self._persona()),
            patch.object(basic_cli, "identity_from_persona", return_value=MagicMock(agent_id="test")),
        ):
            basic_cli.main()
        return session

    def _run_voice(self, read_side_effect) -> MagicMock:
        session = self._fake_session()
        with (
            patch.dict(os.environ, {"Guga_DEBUG": "0"}, clear=False),
            patch.object(voice_cli, "_load_env_file"),
            patch.object(voice_cli, "read_user_input", side_effect=read_side_effect),
            patch.object(voice_cli, "create_chat_model", return_value=MagicMock()),
            patch.object(voice_cli, "MemoryManager", return_value=MagicMock()),
            patch.object(voice_cli, "ChatSession", return_value=session),
            patch.object(voice_cli.PersonaManager, "load", return_value=self._persona()),
            patch.object(voice_cli, "identity_from_persona", return_value=MagicMock(agent_id="test")),
            patch.object(voice_cli, "configure_voice_tool_mode", return_value=False),
            patch.object(
                voice_cli.GptSoVitsConfig,
                "from_env",
                return_value=MagicMock(endpoint="test", ref_audio_path="test"),
            ),
            patch.object(voice_cli, "GptSoVitsHttpClient", return_value=MagicMock()),
            patch.object(voice_cli, "_prewarm_tts"),
        ):
            voice_cli.main()
        return session

    def test_basic_cli_eof_settles_once(self) -> None:
        session = self._run_basic(EOFError)
        session.settle_memory_for_shutdown.assert_called_once_with()

    def test_basic_cli_exit_settles_once(self) -> None:
        session = self._run_basic(["/exit"])
        session.settle_memory_for_shutdown.assert_called_once_with()

    def test_voice_cli_input_interrupt_settles_once(self) -> None:
        session = self._run_voice(KeyboardInterrupt)
        session.settle_memory_for_shutdown.assert_called_once_with()

    def test_voice_cli_exit_settles_once(self) -> None:
        session = self._run_voice(["/exit"])
        session.settle_memory_for_shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
