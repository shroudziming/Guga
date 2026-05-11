from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class Persona:
    name: str
    system_prompt: str
    description: str = ""


@dataclass
class MemoryHit:
    id: str
    summary: str
    raw_excerpt: str = ""
    score: float = 0.0
    memory_type: str = "episodic"
    source_session_id: str = ""
    source_message_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    last_recalled_at: str = ""
    memory_strength: int = 1
    retention: float = 1.0
    importance: float = 0.0
    confidence: float = 0.0


@dataclass
class DocumentHit:
    chunk_id: str
    text: str
    score: float
    source_id: str
    source_path: str = ""
    created_at: str = ""


@dataclass
class MemoryContext:
    archival_memories: list[str] = field(default_factory=list)
    hits: list[MemoryHit] = field(default_factory=list)
    document_hits: list[DocumentHit] = field(default_factory=list)
    event_summaries: list[MemoryHit] = field(default_factory=list)
    user_portrait: str = ""
