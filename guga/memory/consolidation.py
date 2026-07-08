from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConsolidationConfig:
    batch_turns: int = 10
    include_guga_reflection: bool = True
    enable_archival_updates: bool = True
    enable_profile_updates: bool = True
    enable_personality_updates: bool = True
    max_packet_chars: int = 60000

    def normalized(self) -> "MemoryConsolidationConfig":
        return MemoryConsolidationConfig(
            batch_turns=max(1, int(self.batch_turns or 10)),
            include_guga_reflection=bool(self.include_guga_reflection),
            enable_archival_updates=bool(self.enable_archival_updates),
            enable_profile_updates=bool(self.enable_profile_updates),
            enable_personality_updates=bool(self.enable_personality_updates),
            max_packet_chars=max(4096, int(self.max_packet_chars or 60000)),
        )
