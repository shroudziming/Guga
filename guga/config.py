from pathlib import Path

from guga.types import GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models_cache"
DEFAULT_PERSONA = "default"


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )
