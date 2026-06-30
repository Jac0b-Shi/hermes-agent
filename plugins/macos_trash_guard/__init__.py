"""Local macOS safety policy: steer destructive shell deletes to Trash."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

_DELETE_CMD = chr(114) + chr(109)
_TRASH_CMD = "trash"

_BLOCK_MESSAGE = (
    f"Blocked: Hermes does not run `{_DELETE_CMD}` on local macOS. "
    f"Use `{_TRASH_CMD} <path>` instead so files are moved to the macOS Trash. "
    f"If `{_TRASH_CMD}` is not installed, install it with `brew install trash`."
)

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATOR_CHARS = set(";|&()\n")
_SUDO_OPTIONS_WITH_ARG = {
    "-u", "-g", "-h", "-p", "-C", "-T", "-t", "-U", "-R", "-r",
    "--user", "--group", "--host", "--prompt", "--chdir",
    "--login-class", "--role", "--type", "--askpass",
}
_ENV_OPTIONS_WITH_ARG = {
    "-u", "--unset", "-S", "--split-string", "-C", "--chdir",
}
_GENERIC_WRAPPERS = {"exec", "time", "noglob", "builtin"}


def _terminal_env_is_local_macos() -> bool:
    return sys.platform == "darwin" and os.getenv("TERMINAL_ENV", "local") == "local"


def _skip_horizontal_space(command: str, index: int) -> int:
    while index < len(command) and command[index].isspace() and command[index] != "\n":
        index += 1
    return index


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell word, respecting basic quotes and escapes."""
    i = start
    out: list[str] = []
    quote: str | None = None
    while i < len(command):
        ch = command[i]
        if quote is None and (ch.isspace() or ch in _SEPARATOR_CHARS):
            break
        out.append(ch)
        if ch == "\\":
            i += 1
            if i < len(command):
                out.append(command[i])
        elif quote is None and ch in {"'", '"'}:
            quote = ch
        elif quote == ch:
            quote = None
        i += 1
    return "".join(out), i


def _normalize_shell_word(token: str) -> str:
    """Collapse simple shell quoting enough to identify command names."""
    result: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(token):
        ch = token[i]
        if ch == "\\":
            i += 1
            if i < len(token):
                result.append(token[i])
        elif quote is None and ch in {"'", '"'}:
            quote = ch
        elif quote == ch:
            quote = None
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _command_basename(token: str) -> str:
    word = _normalize_shell_word(token).strip()
    return word.rsplit("/", 1)[-1]


def _looks_like_env_assignment(token: str) -> bool:
    return bool(_ENV_ASSIGNMENT_RE.match(_normalize_shell_word(token)))


def _skip_option_arg_if_needed(command: str, index: int, option: str, options_with_arg: set[str]) -> int:
    if "=" in option:
        return index
    needs_arg = option in options_with_arg
    if not needs_arg and option.startswith("--"):
        needs_arg = option in options_with_arg
    if not needs_arg:
        return index
    index = _skip_horizontal_space(command, index)
    if index >= len(command) or command[index] in _SEPARATOR_CHARS:
        return index
    _, index = _read_shell_token(command, index)
    return index


def _skip_sudo_prefix(command: str, index: int) -> int:
    index = _skip_horizontal_space(command, index)
    while index < len(command):
        token, next_index = _read_shell_token(command, index)
        word = _normalize_shell_word(token)
        if word == "--":
            return _skip_horizontal_space(command, next_index)
        if not word.startswith("-"):
            return index
        index = _skip_option_arg_if_needed(command, next_index, word, _SUDO_OPTIONS_WITH_ARG)
        index = _skip_horizontal_space(command, index)
    return index


def _skip_env_prefix(command: str, index: int) -> int:
    index = _skip_horizontal_space(command, index)
    while index < len(command):
        token, next_index = _read_shell_token(command, index)
        word = _normalize_shell_word(token)
        if word == "--":
            index = _skip_horizontal_space(command, next_index)
            continue
        if word.startswith("-"):
            index = _skip_option_arg_if_needed(command, next_index, word, _ENV_OPTIONS_WITH_ARG)
            index = _skip_horizontal_space(command, index)
            continue
        if _looks_like_env_assignment(token):
            index = _skip_horizontal_space(command, next_index)
            continue
        return index
    return index


def _skip_command_wrapper_prefix(command: str, index: int) -> int | None:
    """Return the next command index, or None for informational `command -v/-V`."""
    index = _skip_horizontal_space(command, index)
    while index < len(command):
        token, next_index = _read_shell_token(command, index)
        word = _normalize_shell_word(token)
        if word in {"-v", "-V"}:
            return None
        if word == "--" or not word.startswith("-"):
            return index
        index = _skip_horizontal_space(command, next_index)
    return index


def _read_effective_command_name(command: str, index: int) -> tuple[str, int]:
    """Return the command name after common wrappers such as sudo/env/exec."""
    index = _skip_horizontal_space(command, index)
    for _ in range(12):
        if index >= len(command) or command[index] in _SEPARATOR_CHARS:
            return "", index
        token, next_index = _read_shell_token(command, index)
        if not token:
            return "", max(next_index, index + 1)
        if _looks_like_env_assignment(token):
            index = _skip_horizontal_space(command, next_index)
            continue
        name = _command_basename(token)
        if name == "sudo":
            index = _skip_sudo_prefix(command, next_index)
            continue
        if name == "env":
            index = _skip_env_prefix(command, next_index)
            continue
        if name == "command":
            unwrapped = _skip_command_wrapper_prefix(command, next_index)
            if unwrapped is None:
                return name, next_index
            index = unwrapped
            continue
        if name in _GENERIC_WRAPPERS:
            index = _skip_horizontal_space(command, next_index)
            continue
        return name, next_index
    return "", index


def _contains_delete_command_invocation(command: str) -> bool:
    if not isinstance(command, str) or not command.strip():
        return False

    i = 0
    command_start = True
    while i < len(command):
        ch = command[i]
        if ch.isspace():
            if ch == "\n":
                command_start = True
            i += 1
            continue
        if ch == "#" and command_start:
            newline = command.find("\n", i)
            if newline == -1:
                return False
            i = newline + 1
            command_start = True
            continue
        if ch == "$" and i + 1 < len(command) and command[i + 1] == "(":
            i += 2
            command_start = True
            continue
        if ch == "`":
            i += 1
            command_start = True
            continue
        if ch in _SEPARATOR_CHARS:
            command_start = ch != ")"
            i += 1
            continue
        if command_start:
            name, next_index = _read_effective_command_name(command, i)
            if name == _DELETE_CMD:
                return True
            i = max(next_index, i + 1)
            command_start = False
            continue
        _, next_index = _read_shell_token(command, i)
        i = max(next_index, i + 1)
    return False


def _pre_tool_call(tool_name: str = "", args: dict[str, Any] | None = None, **_: Any) -> dict[str, str] | None:
    if tool_name != "terminal" or not _terminal_env_is_local_macos():
        return None
    command = (args or {}).get("command")
    if _contains_delete_command_invocation(command):
        return {"action": "block", "message": _BLOCK_MESSAGE}
    return None


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", _pre_tool_call)
