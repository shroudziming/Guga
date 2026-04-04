from pathlib import Path

from guga.config import PROJECT_ROOT


def personas_dir() -> Path:
    return PROJECT_ROOT / "config" / "personas"


def user_profile_file() -> Path:
    return PROJECT_ROOT / "config" / "user_profile.json"
