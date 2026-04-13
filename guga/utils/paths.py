from pathlib import Path

from guga.config import PROJECT_ROOT


def personas_dir() -> Path:
    return PROJECT_ROOT / "config" / "personas"


def user_profile_file() -> Path:
    return profile_file()


def memory_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "memory"


def sessions_dir() -> Path:
    return memory_data_dir() / "sessions"


def profile_file() -> Path:
    return memory_data_dir() / "profile.json"


def relationship_state_file() -> Path:
    return memory_data_dir() / "relationship_state.json"


def core_memory_file() -> Path:
    return memory_data_dir() / "core_memory.jsonl"


def archival_memory_file() -> Path:
    return memory_data_dir() / "archival_memory.jsonl"


def memory_indexes_dir() -> Path:
    return memory_data_dir() / "indexes"
