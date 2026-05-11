from __future__ import annotations

import unittest

from guga.models.openai_compatible_chat_model import ApiConfig, OpenAICompatibleChatModel


class OpenAICompatibleChatModelTest(unittest.TestCase):
    def test_extract_text_content_ignores_none_stream_chunks(self) -> None:
        model = OpenAICompatibleChatModel(
            model_id="fake",
            api_config=ApiConfig(base_url="https://example.invalid", api_key="fake"),
        )

        self.assertEqual(model._extract_text_content(None), "")


if __name__ == "__main__":
    unittest.main()
