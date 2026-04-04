from __future__ import annotations

import json
from pathlib import Path

from guga.types import Persona


class PersonaManager:
    def __init__(self, personas_dir: Path) -> None:
        self.personas_dir = personas_dir

    def load(self, persona_name: str) -> Persona:
        file_path = self.personas_dir / f"{persona_name}.json"
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return Persona(
            name=data["name"],
            system_prompt=data["system_prompt"],
            description=data.get("description", ""),
        )
