from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructuredReply:
    content: str
    finish_reason: str = "unknown"
    response_mode: str = "json_object"
    output_chars: int = 0
    usage: dict[str, int] = field(default_factory=dict)
