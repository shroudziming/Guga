from __future__ import annotations

import re
from dataclasses import dataclass

from guga.types import Persona
from guga.utils.paths import memory_data_dir


_AGENT_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,63})$")


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    reflection_context: str
    persona_source: str
    persona_fingerprint: str

    def __post_init__(self) -> None:
        agent_id = self.agent_id.strip()
        if not _AGENT_ID_PATTERN.fullmatch(agent_id):
            raise ValueError(f"unsafe agent_id: {self.agent_id!r}")
        if not self.reflection_context.strip():
            raise ValueError("reflection_context must not be empty")
        if not self.persona_source.strip():
            raise ValueError("persona_source must not be empty")
        if not self.persona_fingerprint.strip():
            raise ValueError("persona_fingerprint must not be empty")
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "reflection_context", self.reflection_context.strip())
        object.__setattr__(self, "persona_source", self.persona_source.strip())
        object.__setattr__(self, "persona_fingerprint", self.persona_fingerprint.strip())


def agent_memory_root(agent_id: str) -> "Path":
    from pathlib import Path

    validated = AgentIdentity(
        agent_id=agent_id,
        reflection_context="agent-root",
        persona_source="agent-root",
        persona_fingerprint="agent-root",
    ).agent_id
    return Path(memory_data_dir()) / "agents" / validated


def identity_from_persona(persona: Persona) -> AgentIdentity:
    return AgentIdentity(
        agent_id=persona.agent_id or persona.name,
        reflection_context=persona.reflection_context or persona.description or persona.name,
        persona_source=persona.source_path or f"persona:{persona.name}",
        persona_fingerprint=persona.persona_fingerprint or persona.name,
    )
