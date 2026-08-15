from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TextIO


def read_user_input(prompt: str) -> str:
    """Read one line without Windows console's previous-line completion."""
    if os.name != "nt" or not sys.stdin.isatty():
        return input(prompt)

    import msvcrt

    return read_console_line(prompt, read_key=msvcrt.getwch, output=sys.stdout)


def read_console_line(
    prompt: str,
    *,
    read_key: Callable[[], str],
    output: TextIO,
) -> str:
    buffer: list[str] = []
    cursor = 0
    output.write(prompt)
    output.flush()

    while True:
        key = read_key()
        if key == "\r":
            output.write("\n")
            output.flush()
            return "".join(buffer)
        if key == "\x03":
            raise KeyboardInterrupt
        if key == "\x1a":
            raise EOFError
        if key in ("\x00", "\xe0"):
            code = read_key()
            if code == "K" and cursor > 0:  # left
                cursor -= 1
                output.write("\b")
            elif code == "M" and cursor < len(buffer):  # right
                output.write(buffer[cursor])
                cursor += 1
            elif code == "G":  # home
                output.write("\b" * cursor)
                cursor = 0
            elif code == "O":  # end
                output.write("".join(buffer[cursor:]))
                cursor = len(buffer)
            elif code == "S" and cursor < len(buffer):  # delete
                del buffer[cursor]
                suffix = "".join(buffer[cursor:])
                output.write(suffix + " " + "\b" * (len(suffix) + 1))
            output.flush()
            continue
        if key == "\b":
            if cursor > 0:
                cursor -= 1
                del buffer[cursor]
                suffix = "".join(buffer[cursor:])
                output.write("\b" + suffix + " " + "\b" * (len(suffix) + 1))
                output.flush()
            continue
        if key < " ":
            continue

        buffer.insert(cursor, key)
        cursor += 1
        suffix = "".join(buffer[cursor:])
        output.write(key + suffix + "\b" * len(suffix))
        output.flush()
