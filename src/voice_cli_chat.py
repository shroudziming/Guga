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
from guga.persona import PersonaManager
from guga.utils.debug_reporter import FileDebugSink
from guga.utils.paths import debug_reports_dir, personas_dir
from guga.voice import (
    GptSoVitsConfig,
    GptSoVitsHttpClient,
    VoiceChatRunner,
    audio_player_from_env,
    configure_voice_tool_mode,
    prewarm_tts_client,
    sentence_buffer_from_env,
)


def _load_env_file() -> None:
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
    tools_enabled = configure_voice_tool_mode(os.environ)

    tts_config = GptSoVitsConfig.from_env()

    print("[Guga] 语音 CLI 聊天")
    print("命令: /clear 清空会话, /rag_rebuild 重建RAG索引, /exit 退出")
    print("提示: 生成中按 Ctrl+C 可停止输出和后续语音")
    print(f"model={model_id}")
    print(f"persona={persona_name}")
    print(f"voice_tools={'on' if tools_enabled else 'off'}")
    print(f"tts_endpoint={tts_config.endpoint}")
    print(f"tts_ref_audio={tts_config.ref_audio_path}\n")
    persona = PersonaManager(personas_dir()).load(persona_name)
    agent_identity = identity_from_persona(persona)
    if debug_enabled:
        print("[DEBUG] 交互调试已开启（可用 Guga_DEBUG=0 关闭）\n")
        print(f"[DEBUG] 报告目录: {debug_reports_dir(agent_identity.agent_id)}\n")

    sink = FileDebugSink(debug_reports_dir(agent_identity.agent_id)) if debug_enabled else None
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

    tts_client = GptSoVitsHttpClient(tts_config)
    _prewarm_tts(tts_client)

    while True:
        user_text = input("你> ").strip()
        if not user_text:
            continue

        if user_text == "/exit":
            result = session.settle_memory_for_shutdown()
            print(f"记忆整理: {result.get('status', 'unknown')}")
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
        print("小咕嘎> ", end="", flush=True)
        player = audio_player_from_env(os.environ)
        runner = VoiceChatRunner(
            session=session,
            tts_client=tts_client,
            audio_player=player,
            text_sink=lambda chunk: print(chunk, end="", flush=True),
            sentence_buffer=sentence_buffer_from_env(os.environ),
            raise_tts_errors=False,
            expression_tags=persona.expression_tags,
        )

        try:
            summary = runner.run_turn(user_text, cancel_event=cancel_event)
            print("\n")
            _print_metrics(summary)
        except KeyboardInterrupt:
            cancel_event.set()
            print("\n[已停止生成]\n")


def _print_metrics(summary) -> None:
    first_audio = "n/a" if summary.first_audio_ms is None else f"{summary.first_audio_ms}ms"
    average_rtf = "n/a" if summary.average_rtf is None else f"{summary.average_rtf:.2f}"
    total = "n/a" if summary.total_ms is None else f"{summary.total_ms}ms"
    print(
        "[voice] "
        f"sentences={summary.sentences} "
        f"first_text={_format_ms(summary.first_text_ms)} "
        f"first_sentence={_format_ms(summary.first_sentence_ms)} "
        f"first_audio={first_audio} "
        f"rtf={average_rtf} "
        f"audio={summary.audio_seconds:.2f}s "
        f"tts={summary.tts_seconds:.2f}s "
        f"total={total}\n"
    )


def _prewarm_tts(tts_client) -> None:
    result = prewarm_tts_client(tts_client, os.environ)
    if result.status == "disabled":
        print("[voice] tts_prewarm=disabled\n")
        return
    if result.ok:
        print(f"[voice] tts_prewarm=ok elapsed={result.elapsed_seconds:.2f}s\n")
        return
    print(f"[voice] tts_prewarm=failed elapsed={result.elapsed_seconds:.2f}s error={result.error}\n")


def _format_ms(value: int | None) -> str:
    return "n/a" if value is None else f"{value}ms"


if __name__ == "__main__":
    main()
