from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaText:
    text: str


@dataclass(frozen=True)
class PersonaExpression:
    tag: str


class PersonaOutputParser:
    def __init__(self, tags: tuple[str, ...]) -> None:
        self._tags = frozenset(tags)
        self._buffer = ""

    def feed(self, chunk: str) -> list[PersonaText | PersonaExpression]:
        remaining = self._buffer + chunk
        self._buffer = ""
        events: list[PersonaText | PersonaExpression] = []

        while remaining:
            bracket_start = remaining.find("[")
            if bracket_start < 0:
                events.append(PersonaText(remaining))
                break
            if bracket_start:
                events.append(PersonaText(remaining[:bracket_start]))
                remaining = remaining[bracket_start:]

            bracket_end = remaining.find("]", 1)
            nested_bracket = remaining.find("[", 1)
            if nested_bracket >= 0 and (
                bracket_end < 0 or nested_bracket < bracket_end
            ):
                events.append(PersonaText(remaining[:nested_bracket]))
                remaining = remaining[nested_bracket:]
                continue
            if bracket_end >= 0:
                tag = remaining[1:bracket_end]
                if tag in self._tags:
                    events.append(PersonaExpression(tag))
                else:
                    events.append(PersonaText(remaining[: bracket_end + 1]))
                remaining = remaining[bracket_end + 1 :]
                continue

            partial_tag = remaining[1:]
            if any(tag.startswith(partial_tag) for tag in self._tags):
                self._buffer = remaining
            else:
                events.append(PersonaText(remaining))
            break

        return events

    def flush(self) -> list[PersonaText]:
        if not self._buffer:
            return []
        text = self._buffer
        self._buffer = ""
        return [PersonaText(text)]
