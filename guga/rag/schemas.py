from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DocumentChunk:
    id: str
    text: str
    source_type: str
    source_id: str
    source_path: str = ""
    source_session_id: str = ""
    source_message_id: str = ""
    created_at: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DocumentChunk":
        return cls(
            id=str(payload.get("id", "")),
            text=str(payload.get("text", "")),
            source_type=str(payload.get("source_type", "")),
            source_id=str(payload.get("source_id", "")),
            source_path=str(payload.get("source_path", "")),
            source_session_id=str(payload.get("source_session_id", "")),
            source_message_id=str(payload.get("source_message_id", "")),
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    source_type: str
    source_id: str
    source_path: str = ""
    source_session_id: str = ""
    source_message_id: str = ""
    created_at: str = ""
