from __future__ import annotations


class SpokenTextFilter:
    """Remove text that should be displayed but not spoken."""

    def __init__(self) -> None:
        self._paren_depth = 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        output: list[str] = []
        for char in chunk:
            if char in {"(", "（"}:
                self._paren_depth += 1
                continue

            if char in {")", "）"}:
                if self._paren_depth > 0:
                    self._paren_depth -= 1
                    continue

            if self._paren_depth == 0:
                output.append(char)

        return "".join(output)
