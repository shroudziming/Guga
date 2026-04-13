from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MessageRecord:
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    source: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageRecord":
        return cls(
            id=str(data.get("id", "")),
            session_id=str(data.get("session_id", "")),
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            created_at=str(data.get("created_at", "")),
            source=str(data.get("source", "chat")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ProfileRecord:
    user_id: str = "default_user"
    display_name: str | None = None
    preferred_name: str | None = None
    communication_preferences: dict[str, Any] = field(default_factory=dict)
    stable_preferences: dict[str, Any] = field(default_factory=dict)
    life_context: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileRecord":
        return cls(
            user_id=str(data.get("user_id", "default_user")),
            display_name=data.get("display_name"),
            preferred_name=data.get("preferred_name"),
            communication_preferences=dict(data.get("communication_preferences", {})),
            stable_preferences=dict(data.get("stable_preferences", {})),
            life_context=dict(data.get("life_context", {})),
            updated_at=str(data.get("updated_at", "")),
            version=int(data.get("version", 1)),
        )


@dataclass
class CoreMemoryRecord:
    id: str
    kind: str
    text: str
    importance: float
    confidence: float
    source_message_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    last_accessed_at: str | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoreMemoryRecord":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            text=str(data.get("text", "")),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            source_message_ids=list(data.get("source_message_ids", [])),
            created_at=str(data.get("created_at", "")),
            last_accessed_at=data.get("last_accessed_at"),
            status=str(data.get("status", "active")),
        )


@dataclass
class ArchivalMemoryRecord:
    id: str
    type: str
    topic: str
    summary: str
    raw_excerpt: str
    emotion: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.5
    created_at: str = ""
    event_time_start: str | None = None
    event_time_end: str | None = None
    last_accessed_at: str | None = None
    access_count: int = 0
    source_session_id: str = ""
    source_message_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchivalMemoryRecord":
        return cls(
            id=str(data.get("id", "")),
            type=str(data.get("type", "episodic")),
            topic=str(data.get("topic", "general")),
            summary=str(data.get("summary", "")),
            raw_excerpt=str(data.get("raw_excerpt", "")),
            emotion=list(data.get("emotion", [])),
            entities=list(data.get("entities", [])),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            created_at=str(data.get("created_at", "")),
            event_time_start=data.get("event_time_start"),
            event_time_end=data.get("event_time_end"),
            last_accessed_at=data.get("last_accessed_at"),
            access_count=int(data.get("access_count", 0)),
            source_session_id=str(data.get("source_session_id", "")),
            source_message_ids=list(data.get("source_message_ids", [])),
            tags=list(data.get("tags", [])),
            status=str(data.get("status", "active")),
        )


@dataclass
class RelationshipState:
    user_id: str = "default_user"
    relationship_stage: str = "new"
    interaction_style_hints: list[str] = field(default_factory=list)
    sensitivity_topics: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContext:
    profile_summary: str = ""
    relationship_hints: list[str] = field(default_factory=list)
    core_memories: list[str] = field(default_factory=list)
    archival_memories: list[str] = field(default_factory=list)
    session_summary: str = ""
