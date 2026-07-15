from __future__ import annotations

import unittest

from guga.persona import (
    PersonaExpression,
    PersonaOutputParser,
    PersonaText,
)


class PersonaOutputParserTest(unittest.TestCase):
    def test_parses_complete_tag_between_visible_text(self) -> None:
        parser = PersonaOutputParser(("happy", "side"))

        self.assertEqual(
            parser.feed("你好[happy]世界"),
            [PersonaText("你好"), PersonaExpression("happy"), PersonaText("世界")],
        )

    def test_parses_tag_split_across_stream_chunks(self) -> None:
        parser = PersonaOutputParser(("happy", "side"))

        self.assertEqual(parser.feed("[hap"), [])
        self.assertEqual(
            parser.feed("py]你好。"),
            [PersonaExpression("happy"), PersonaText("你好。")],
        )

    def test_parses_repeated_adjacent_tags(self) -> None:
        parser = PersonaOutputParser(("happy", "side"))

        self.assertEqual(
            parser.feed("[happy][happy][side]"),
            [
                PersonaExpression("happy"),
                PersonaExpression("happy"),
                PersonaExpression("side"),
            ],
        )

    def test_unknown_tag_is_visible_text(self) -> None:
        parser = PersonaOutputParser(("happy",))

        self.assertEqual(
            parser.feed("[unknown]你好"),
            [PersonaText("[unknown]"), PersonaText("你好")],
        )

    def test_invalidated_partial_tag_does_not_consume_following_valid_tag(self) -> None:
        parser = PersonaOutputParser(("happy", "side"))

        self.assertEqual(parser.feed("[hap"), [])
        self.assertEqual(
            parser.feed("x[side]"),
            [PersonaText("[hapx"), PersonaExpression("side")],
        )

    def test_flush_preserves_unterminated_bracket_text(self) -> None:
        parser = PersonaOutputParser(("happy",))

        self.assertEqual(parser.feed("[hap"), [])
        self.assertEqual(parser.flush(), [PersonaText("[hap")])
        self.assertEqual(parser.flush(), [])


if __name__ == "__main__":
    unittest.main()
