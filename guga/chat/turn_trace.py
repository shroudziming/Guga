from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guga.tools import ToolCall


@dataclass
class TurnTrace:
    turn_id: str
    user_message_id: str
    reference_time: str
    time_queries: list[dict[str, Any]] = field(default_factory=list)
    tool_interactions: list[dict[str, Any]] = field(default_factory=list)

    def record_tool(self, call: ToolCall, result: dict[str, Any]) -> None:
        returned_ids = [
            str(match.get("record_id", ""))
            for group in (result.get("queries", []) or [])
            if isinstance(group, dict)
            for match in (group.get("matches", []) or [])
            if isinstance(match, dict) and str(match.get("record_id", ""))
        ]
        self.tool_interactions.append(
            {
                "tool_call_id": call.id,
                "tool": call.name,
                "arguments": dict(call.arguments),
                "ok": bool(result.get("ok")),
                "returned_memory_ids": returned_ids,
                "error": (
                    dict(result.get("error", {}))
                    if isinstance(result.get("error"), dict)
                    else str(result.get("error", ""))
                ),
            }
        )
        if call.name == "memory_time_search" and result.get("ok") is True:
            self.time_queries.extend(
                dict(group["query"])
                for group in (result.get("queries", []) or [])
                if isinstance(group, dict) and isinstance(group.get("query"), dict)
            )

    def as_metadata(self, assistant_message_id: str) -> dict[str, Any]:
        turn: dict[str, Any] = {
            "turn_id": self.turn_id,
            "assistant_message_id": assistant_message_id,
            "tool_interactions": list(self.tool_interactions),
        }
        if self.time_queries:
            turn["temporal_context"] = {
                "source_message_id": self.user_message_id,
                "reference_time": self.reference_time,
                "time_queries": list(self.time_queries),
            }
        return {"turn": turn}
