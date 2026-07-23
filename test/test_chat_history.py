from __future__ import annotations

import unittest

from guga.chat.history import ChatHistory


class ChatHistoryTest(unittest.TestCase):
    def test_default_history_retains_45_complete_turns(self) -> None:
        history = ChatHistory()

        for turn in range(46):
            history.add_user(f"user-{turn}")
            history.add_assistant(f"assistant-{turn}")

        messages = history.as_messages()
        self.assertEqual(len(messages), 90)
        self.assertEqual(messages[0]["content"], "user-1")
        self.assertEqual(messages[-1]["content"], "assistant-45")


if __name__ == "__main__":
    unittest.main()
