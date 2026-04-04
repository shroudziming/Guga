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
from guga.models import create_chat_model
from guga.persona import PersonaManager
from guga.utils.paths import personas_dir


def main() -> None:
    model_id = os.environ.get("Guga_MODEL_ID", DEFAULT_MODEL_ID)
    cache_dir = os.environ.get("Guga_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    persona_name = os.environ.get("Guga_PERSONA", "default")

    print("[Guga] 多轮 CLI 聊天")
    print("命令: /clear 清空会话, /exit 退出")
    print("提示: 生成中按 Ctrl+C 可停止输出")
    print(f"model={model_id}")
    print(f"persona={persona_name}\n")

    persona = PersonaManager(personas_dir()).load(persona_name)
    model = create_chat_model(model_id=model_id, cache_dir=cache_dir)
    session = ChatSession(
        model=model,
        system_prompt=persona.system_prompt,
        generation=default_generation_config(),
        max_turns=10,
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

        cancel_event = Event()
        stream = session.reply_stream(user_text, cancel_event=cancel_event)

        print("小咕嘎> ", end="", flush=True)
        try:
            for chunk in stream:
                print(chunk, end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            cancel_event.set()
            for _ in stream:
                pass
            print("\n[已停止生成]\n")


if __name__ == "__main__":
    main()
