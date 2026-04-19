from pathlib import Path

from guga.config import PROJECT_ROOT


def personas_dir() -> Path:
    return PROJECT_ROOT / "config" / "personas"


def user_profile_file() -> Path:
    return PROJECT_ROOT / "config" / "user_profile.json"


def memory_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "memory"


def debug_reports_dir() -> Path:
    return memory_data_dir() / "debug_reports"


def rag_data_dir() -> Path:
    return memory_data_dir() / "rag"


def rag_index_dir() -> Path:
    return rag_data_dir() / "index"


def rag_documents_dir() -> Path:
    return PROJECT_ROOT / "data" / "documents"
