from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


_EXPRESSION_TAGS = {
    "normal",
    "angry",
    "sided_angry",
    "blush",
    "sided_blush",
    "happy",
    "sad",
    "surprised",
    "sided_surprised",
    "side",
    "sided_thinking",
    "annoyed",
    "sided_worried",
    "eyes_closed",
    "sided_eyes_closed",
    "sided_pleasant",
    "disappointed",
    "indifferent",
    "pissed",
    "winking",
}


@dataclass(frozen=True)
class LoadedPersonaSkill:
    body: str
    expression_tags: tuple[str, ...]
    source_path: str
    fingerprint: str


def load_persona_skill(path: Path, personas_dir: Path) -> LoadedPersonaSkill:
    root = personas_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Skill path escapes personas directory: {resolved}")
    raw = resolved.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    if set(frontmatter) != {"name", "description"}:
        raise ValueError("Skill frontmatter must contain exactly name and description")
    body = body.strip()
    if not body:
        raise ValueError("empty Skill body")
    tags = _extract_expression_tags(body)
    source = resolved.relative_to(root.parents[1]).as_posix()
    return LoadedPersonaSkill(
        body=body,
        expression_tags=tags,
        source_path=source,
        fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    normalized = raw.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError("Skill must start with frontmatter")
    parts = normalized.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("Skill frontmatter is not closed")
    metadata: dict[str, str] = {}
    for line in parts[0][4:].splitlines():
        if not line.strip() or ":" not in line:
            raise ValueError("Skill frontmatter must use simple key-value fields")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key or not value or key in metadata:
            raise ValueError("Skill frontmatter contains an invalid field")
        metadata[key] = value
    return metadata, parts[1]


def _extract_expression_tags(body: str) -> tuple[str, ...]:
    lines = [line for line in body.splitlines() if line.startswith("固定标签清单：")]
    if len(lines) != 1:
        raise ValueError("Skill must contain exactly one fixed expression tag list")
    code_spans = re.findall(r"`([^`]*)`", lines[0])
    if len(code_spans) != 1:
        raise ValueError("fixed expression tags must be backtick-delimited")
    tags = tuple(code_spans[0].split())
    if len(tags) != len(set(tags)):
        raise ValueError("duplicate expression tag")
    if set(tags) != _EXPRESSION_TAGS or len(tags) != len(_EXPRESSION_TAGS):
        raise ValueError("Skill must declare the exact allowed expression tags")
    return tags
