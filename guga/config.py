import os
from pathlib import Path

from guga.types import GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models_cache"
DEFAULT_PERSONA = "default"
DEFAULT_MEMORY_TOP_K = 4
DEFAULT_DOCUMENT_TOP_K = 4
DEFAULT_MEMORY_RECENCY_WEIGHT = 0.2
DEFAULT_CURRENT_TURN_SCORE_FACTOR = 0.2
DEFAULT_MEMORY_MIN_SCORE = 0.15
DEFAULT_MEMORY_DECAY_ENABLED = False
DEFAULT_MEMORY_DECAY_THRESHOLD = 0.05
DEFAULT_MEMORY_DECAY_MIN_AGE_DAYS = 365.0
DEFAULT_RAG_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RAG_CHUNK_SIZE = 220
DEFAULT_RAG_CHUNK_OVERLAP = 40
DEFAULT_RAG_ENABLE_SEMANTIC = True


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=_env_int("Guga_MAX_NEW_TOKENS", 1024, minimum=64),
        temperature=_env_float("Guga_TEMPERATURE", 0.7),
        top_p=_env_float("Guga_TOP_P", 0.9),
    )
