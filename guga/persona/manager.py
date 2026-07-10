from __future__ import annotations

import hashlib
import json
from pathlib import Path

from guga.types import Persona


class PersonaManager:
    def __init__(self, personas_dir: Path) -> None:
        self.personas_dir = personas_dir

    def load(self, persona_name: str) -> Persona:
        file_path = self.personas_dir / f"{persona_name}.json"
        data = json.loads(file_path.read_text(encoding="utf-8"))
        reflection_context = str(data.get("reflection_context", "")).strip() or str(data.get("description", "")).strip() or data["name"]
        persona_source = self._persona_source(file_path)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "system_prompt": data["system_prompt"],
                    "reflection_context": reflection_context,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return Persona(
            name=data["name"],
            system_prompt=data["system_prompt"],
            description=data.get("description", ""),
            agent_id=str(data.get("agent_id", "")).strip() or data["name"],
            reflection_context=reflection_context,
            source_path=persona_source,
            persona_fingerprint=fingerprint,
        )

    def _persona_source(self, file_path: Path) -> str:
        try:
            return file_path.relative_to(self.personas_dir.parents[1]).as_posix()
        except ValueError:
            return file_path.as_posix()
