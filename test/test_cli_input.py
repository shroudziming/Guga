from __future__ import annotations

import io
import unittest

from guga.cli_input import read_console_line


class CliInputTest(unittest.TestCase):
    def test_right_arrow_at_end_does_not_restore_previous_input(self) -> None:
        keys = iter(["新", "\xe0", "M", "\r"])
        output = io.StringIO()

        result = read_console_line("你> ", read_key=lambda: next(keys), output=output)

        self.assertEqual(result, "新")


if __name__ == "__main__":
    unittest.main()
