from pathlib import Path

from guga.types import GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models_cache"
DEFAULT_PERSONA = "default"
DEFAULT_MEMORY_TOP_K = 4
DEFAULT_DOCUMENT_TOP_K = 4
DEFAULT_MEMORY_RECENCY_WEIGHT = 0.2
DEFAULT_RAG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RAG_CHUNK_SIZE = 220
DEFAULT_RAG_CHUNK_OVERLAP = 40
DEFAULT_RAG_ENABLE_SEMANTIC = True


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )
