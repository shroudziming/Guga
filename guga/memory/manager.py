from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from guga.memory.archival_store import ArchivalStore
from guga.memory.clock import now_iso, today_bucket
from guga.memory.core_memory_store import CoreMemoryStore
from guga.memory.profile_store import ProfileStore
from guga.memory.schema import ArchivalMemoryRecord, MemoryContext, MessageRecord
from guga.memory.session_store import SessionStore
from guga.memory.storage import ensure_dir
from guga.utils.paths import memory_data_dir


class MemoryManager:
    def __init__(self, memory_root: Path | None = None) -> None:
        self.memory_root = memory_root or memory_data_dir()
        self.policy = self._load_policy()
        self.session_store = SessionStore(self.memory_root / "sessions")
        self.archival_store = ArchivalStore(self.memory_root / "archival_memory.jsonl")
        self.core_memory_store = CoreMemoryStore(self.memory_root / "core_memory.jsonl")
        self.profile_store = ProfileStore(self.memory_root / "profile.json")
        self._turn_state: dict[str, dict[str, str]] = {}
        self._ensure_base_dirs()

    def prepare_context(self, user_text: str, session_id: str) -> MemoryContext:
        return MemoryContext()

    def record_user_message(self, session_id: str, text: str, source: str = "chat") -> str:
        message_id = self._record_message(session_id=session_id, role="user", text=text, source=source)
        state = self._turn_state.setdefault(session_id, {})
        state["user_message_id"] = message_id
        state["user_text"] = text
        return message_id

    def record_assistant_message(self, session_id: str, text: str, source: str = "chat") -> str:
        message_id = self._record_message(session_id=session_id, role="assistant", text=text, source=source)
        state = self._turn_state.setdefault(session_id, {})
        state["assistant_message_id"] = message_id
        state["assistant_text"] = text
        return message_id

    def finalize_turn(self, session_id: str) -> None:
        state = self._turn_state.get(session_id, {})
        user_text = state.get("user_text", "")

        self._ensure_profile_exists()

        if self._should_archive_user_text(user_text):
            user_message_id = state.get("user_message_id", "")
            archival = self._build_archival_record(
                session_id=session_id,
                user_text=user_text,
                source_message_ids=[user_message_id] if user_message_id else [],
            )
            self.archival_store.append(archival)

        self._turn_state.pop(session_id, None)

    def backup(self, target_dir: str | None = None) -> str:
        target = Path(target_dir) if target_dir else self.memory_root / "backups" / now_iso().replace(":", "-")
        ensure_dir(target)
        return str(target)

    def _ensure_base_dirs(self) -> None:
        ensure_dir(self.memory_root)
        ensure_dir(self.memory_root / "sessions")
        ensure_dir(self.memory_root / "backups")
        ensure_dir(self.memory_root / "indexes")
        ensure_dir(self.memory_root / "embeddings")

    def _load_policy(self) -> dict:
        policy_file = self.memory_root.parent.parent / "config" / "memory" / "memory_policy.json"
        if not policy_file.exists():
            return {}
        return json.loads(policy_file.read_text(encoding="utf-8"))

    def _ensure_profile_exists(self) -> None:
        profile = self.profile_store.load()
        self.profile_store.save(profile)

    def _record_message(self, session_id: str, role: str, text: str, source: str) -> str:
        message_id = f"msg_{uuid4().hex[:10]}"
        record = MessageRecord(
            id=message_id,
            session_id=session_id,
            role=role,
            content=text,
            created_at=now_iso(),
            source=source,
        )
        self.session_store.append(record)
        return message_id

    def _should_archive_user_text(self, user_text: str) -> bool:
        archival_policy = self.policy.get("archival", {})
        min_length = int(archival_policy.get("min_user_text_length", 12))

        if len(user_text) >= min_length:
            return True

        keywords = []
        keywords.extend(list(archival_policy.get("emotion_keywords", [])))
        keywords.extend(list(archival_policy.get("preference_keywords", [])))
        keywords.extend(list(archival_policy.get("identity_keywords", [])))
        return any(keyword in user_text for keyword in keywords)

    def _build_archival_record(
        self,
        session_id: str,
        user_text: str,
        source_message_ids: list[str],
    ) -> ArchivalMemoryRecord:
        archival_policy = self.policy.get("archival", {})
        now = now_iso()
        memory_id = f"mem_{today_bucket().replace('-', '')}_{uuid4().hex[:8]}"
        topic = self._infer_topic(user_text)

        return ArchivalMemoryRecord(
            id=memory_id,
            type="episodic",
            topic=topic,
            summary=f"用户提到：{user_text}",
            raw_excerpt=user_text,
            importance=float(archival_policy.get("default_importance", 0.7)),
            confidence=float(archival_policy.get("default_confidence", 0.7)),
            created_at=now,
            event_time_start=now,
            source_session_id=session_id,
            source_message_ids=source_message_ids,
            tags=[topic],
            status="active",
        )

    def _infer_topic(self, user_text: str) -> str:
        topic_map = {
            "工作": "career",
            "换工作": "career",
            "焦虑": "emotion",
            "压力": "emotion",
            "喜欢": "preference",
            "不喜欢": "preference",
        }
        for keyword, topic in topic_map.items():
            if keyword in user_text:
                return topic
        return "general"
