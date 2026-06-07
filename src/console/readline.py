from __future__ import annotations

import platform
import sys
from typing import Callable

from core import get_logger

log = get_logger("auraderma.cli")


def create_readline(
    model_name: str,
    command_names: list[str],
    command_descriptions: list[str],
    history: list[str],
) -> Callable[[], str | None]:
    """工厂函数：根据操作系统创建合适的 readline 函数。

    - Windows: 使用 msvcrt 实现带下拉补全的交互终端
    - Linux/macOS: 使用标准 input() + readline module 实现基本补全

    Returns:
        一个无参数函数，返回输入字符串或 None（用户退出时）
    """
    system = platform.system()
    log.info("创建 readline: platform=%s", system)

    if system == "Windows":
        return _make_aris_readline(model_name, command_names, command_descriptions, history)
    else:
        return _make_unix_readline(model_name, command_names, history)


# ---------------------------------------------------------------------------
# Windows: msvcrt-based rich readline (preserved from original)
# ---------------------------------------------------------------------------

def _make_aris_readline(
    model_name: str,
    command_names: list[str],
    command_descriptions: list[str],
    history: list[str],
) -> Callable[[], str | None]:
    """ARIS-style readline for Windows using msvcrt."""
    import msvcrt  # Windows only

    command_specs = list(zip(command_names, command_descriptions))

    def readline() -> str | None:
        buf: list[str] = []
        cursor = 0
        sel = 0
        history_idx: int | None = None
        saved_buf: list[str] | None = None

        prompt_str = (
            f"\x1b[36m\x1b[1mAuraDerma \x1b[34m[{model_name}] "
            f"\x1b[33m\x1b[1m> \x1b[0m"
        )
        prompt_visible_len = 15 + len(model_name)

        def _display_width(s: str) -> int:
            width = 0
            for ch in s:
                if (
                    "\u4e00" <= ch <= "\u9fff"
                    or "\u3000" <= ch <= "\u303f"
                    or "\uff00" <= ch <= "\uffef"
                ):
                    width += 2
                else:
                    width += 1
            return width

        def _compute_matches(line: str) -> list[tuple[str, str]]:
            if not line.startswith("/"):
                return []
            text_lower = line.lower()
            out = []
            for name, desc in command_specs:
                it = iter(name.lower())
                if all(c in it for c in text_lower):
                    out.append((name, desc))
            return out

        def _render() -> None:
            nonlocal sel
            line = "".join(buf)
            matches = _compute_matches(line)

            sys.stdout.write("\r\x1b[J")
            sys.stdout.write(prompt_str)
            sys.stdout.write(line)

            if matches:
                max_name = max(len(m[0]) for m in matches)
                name_col = min(max(max_name, 12), 36) + 2
                if sel >= len(matches):
                    sel = len(matches) - 1

                sys.stdout.write("\r\n")
                sys.stdout.write(f"\x1b[2m{'─' * 60}\x1b[0m")
                row_count = 2

                for idx, (name, desc) in enumerate(matches):
                    sys.stdout.write("\r\n")
                    row_count += 1
                    if idx == sel:
                        sys.stdout.write(f"\x1b[1;34m{name}\x1b[0m")
                        sys.stdout.write(" " * (name_col - len(name)))
                        sys.stdout.write(f"\x1b[1;33m{desc}\x1b[0m")
                    else:
                        sys.stdout.write(name)
                        sys.stdout.write(" " * (name_col - len(name)))
                        sys.stdout.write(f"\x1b[33m{desc}\x1b[0m")

                sys.stdout.write(f"\x1b[{row_count - 1}A")

            pre_cursor = "".join(buf[:cursor])
            col = prompt_visible_len + 1 + _display_width(pre_cursor)
            sys.stdout.write(f"\x1b[{col}G")
            sys.stdout.flush()

        _render()

        while True:
            ch = msvcrt.getwch()

            if ch == "\xe0":  # Arrow / function keys
                ch2 = msvcrt.getwch()
                if ch2 == "H":  # Up
                    matches = _compute_matches("".join(buf))
                    if matches:
                        if sel > 0:
                            sel -= 1
                    elif history:
                        if history_idx is None:
                            saved_buf = buf[:]
                            history_idx = len(history) - 1
                        elif history_idx > 0:
                            history_idx -= 1
                        buf = list(history[history_idx])
                        cursor = len(buf)
                        sel = 0
                    _render()
                elif ch2 == "P":  # Down
                    matches = _compute_matches("".join(buf))
                    if matches:
                        if sel < len(_compute_matches("".join(buf))) - 1:
                            sel += 1
                    elif history_idx is not None:
                        if history_idx < len(history) - 1:
                            history_idx += 1
                            buf = list(history[history_idx])
                        else:
                            history_idx = None
                            buf = saved_buf or []
                        cursor = len(buf)
                        sel = 0
                    _render()
                elif ch2 == "K":  # Left
                    if cursor > 0:
                        cursor -= 1
                        _render()
                elif ch2 == "M":  # Right
                    if cursor < len(buf):
                        cursor += 1
                        _render()
                elif ch2 == "G":  # Home
                    cursor = 0
                    _render()
                elif ch2 == "O":  # End
                    cursor = len(buf)
                    _render()
                elif ch2 == "S":  # Delete
                    if cursor < len(buf):
                        buf.pop(cursor)
                        sel = 0
                        _render()
                continue

            if ch == "\r":  # Enter
                sys.stdout.write("\r\x1b[J")
                sys.stdout.write(prompt_str + "".join(buf))
                sys.stdout.write("\n")
                sys.stdout.flush()
                result = "".join(buf)
                if result and result.strip():
                    history.append(result.strip())
                return result

            if ch in ("\b", "\x7f"):  # Backspace
                if cursor > 0:
                    buf.pop(cursor - 1)
                    cursor -= 1
                    sel = 0
                    _render()
                continue

            if ch == "\t":  # Tab
                full = "".join(buf)
                matches = _compute_matches(full)
                if matches and sel < len(matches):
                    buf = list(matches[sel][0])
                    cursor = len(buf)
                    sel = 0
                    _render()
                continue

            if ch == "\x1b":  # Escape
                sel = 0
                _render()
                continue

            if ch in ("\x03", "\x04"):  # Ctrl+C / Ctrl+D
                sys.stdout.write("\r\x1b[J\n")
                sys.stdout.flush()
                return None

            buf.insert(cursor, ch)
            cursor += 1
            sel = 0
            history_idx = None
            saved_buf = None
            _render()

    return readline


# ---------------------------------------------------------------------------
# Unix / cross-platform: readline-based simple input
# ---------------------------------------------------------------------------

def _make_unix_readline(
    model_name: str,
    command_names: list[str],
    history: list[str],
) -> Callable[[], str | None]:
    """Unix-style readline using Python's readline module for history/editing."""
    try:
        import readline  # noqa: F401
    except ImportError:
        pass  # readline not available, fall back to plain input

    prompt = f"AuraDerma [{model_name}] > "

    def readline() -> str | None:
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        cmd = raw.strip()
        if cmd:
            history.append(cmd)
        return cmd

    return readline
