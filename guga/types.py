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
    agent_id: str = ""
    reflection_context: str = ""
    source_path: str = ""
    persona_fingerprint: str = ""


@dataclass
class MemoryHit:
    id: str
    summary: str
    chunk_id: str = ""
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
    day: str = ""
    valid_at: str = ""
    invalid_at: str = ""
    time_source: str = ""
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    score_source: str = ""
    score_components: dict[str, float | str | bool] = field(default_factory=dict)
    is_current_turn: bool = False


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
    user_portrait: str = ""
    query_route: str = "hybrid"
    query_reason: str = ""
