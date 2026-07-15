from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guga.chat import ChatSession
from guga.config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, default_generation_config
from guga.memory.agent_identity import identity_from_persona
from guga.memory.manager import MemoryManager
from guga.models import create_chat_model
from guga.persona import PersonaExpression, PersonaManager, PersonaOutputParser, PersonaText
from guga.utils.debug_reporter import FileDebugSink
from guga.utils.paths import debug_reports_dir, personas_dir


def _load_env_file() -> None:
    """Load PROJECT_ROOT/.env into process env if keys are not already set."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_env_file()

    model_id = os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID)
    cache_dir = os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    persona_name = os.environ.get("Guga_PERSONA", "default")
    debug_enabled = os.environ.get("Guga_DEBUG", "1") != "0"

    print("[Guga] 多轮 CLI 聊天")
    print("命令: /clear 清空会话, /rag_rebuild 重建RAG索引, /exit 退出")
    print("提示: 生成中按 Ctrl+C 可停止输出")
    print(f"model={model_id}")
    print(f"persona={persona_name}\n")
    persona = PersonaManager(personas_dir()).load(persona_name)
    agent_identity = identity_from_persona(persona)
    if debug_enabled:
        print("[DEBUG] 交互调试已开启（可用 Guga_DEBUG=0 关闭）\n")
    sink = FileDebugSink(debug_reports_dir(agent_identity.agent_id)) if debug_enabled else None
    if debug_enabled:
        print(f"[DEBUG] 报告目录: {debug_reports_dir(agent_identity.agent_id)}\n")

    model = create_chat_model(model_id=model_id, cache_dir=cache_dir)
    memory_manager = MemoryManager(
        model=model,
        debug=debug_enabled,
        debug_sink=sink,
        agent_identity=agent_identity,
    )
    session = ChatSession(
        model=model,
        system_prompt=persona.system_prompt,
        generation=default_generation_config(),
        max_turns=10,
        memory_manager=memory_manager,
        debug=debug_enabled,
        debug_sink=sink,
    )

    while True:
        user_text = input("你> ").strip()
        if not user_text:
            continue

        if user_text == "/exit":
            print("已退出。")
            return

        if user_text == "/clear":
            session.clear()
            print("会话已清空。")
            continue

        if user_text == "/rag_rebuild":
            result = session.memory_manager.rebuild_rag_indexes(session_id=session.session_id)
            print(
                f"RAG 索引已重建: memory_chunks={result['memory_chunks']}, "
                f"document_chunks={result['document_chunks']}, total_chunks={result['total_chunks']}"
            )
            continue

        cancel_event = Event()
        stream = session.reply_stream(user_text, cancel_event=cancel_event)
        output_parser = PersonaOutputParser(persona.expression_tags)

        print("小咕嘎> ", end="", flush=True)
        try:
            for chunk in stream:
                _render_persona_events(output_parser.feed(chunk), debug_enabled=debug_enabled)
            _render_persona_events(output_parser.flush(), debug_enabled=debug_enabled)
            print("\n")
        except KeyboardInterrupt:
            cancel_event.set()
            for _ in stream:
                pass
            _render_persona_events(output_parser.flush(), debug_enabled=debug_enabled)
            print("\n[已停止生成]\n")


def _render_persona_events(events, *, debug_enabled: bool) -> None:
    for event in events:
        if isinstance(event, PersonaExpression):
            if debug_enabled:
                print(f"[DEBUG][persona_expression] tag={event.tag}", file=sys.stderr)
            continue
        if isinstance(event, PersonaText):
            print(event.text, end="", flush=True)


if __name__ == "__main__":
    main()
