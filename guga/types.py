from dataclasses import dataclass


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
