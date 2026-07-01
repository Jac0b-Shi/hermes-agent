"""Shared shell-argv helpers — prefix stripping and option-argument consumption.

Used by both the core policy engine providers and the legacy inline
denylist in ``agent/context_acquisition.py``.  Keeping the logic in one
place avoids divergence between the two enforcement layers.
"""
from __future__ import annotations

import os
import shlex
from typing import List

# ── Option-argument sets ───────────────────────────────────────────────

# sudo options that consume a following argument
SUDO_OPTIONS_WITH_ARG: frozenset[str] = frozenset({
    "-u", "-g", "-h", "-p", "-C", "-T", "-t", "-U", "-R", "-r",
    "--user", "--group", "--host", "--prompt", "--chdir",
    "--login-class", "--role", "--type", "--askpass",
})

# env options that consume a following argument.
# NOTE: -S / --split-string are intentionally excluded — their argument
# is a shell command string to be recursively parsed, not a regular value.
ENV_OPTIONS_WITH_ARG: frozenset[str] = frozenset({
    "-u", "--unset", "-C", "--chdir",
})

# env -S / --split-string: the argument is a command string that must be
# recursively split and re-checked against the denylist.
_ENV_SPLIT_STRING_OPTIONS: frozenset[str] = frozenset({"-S", "--split-string"})

# Sentinel returned when env -S parsing fails — treated as a block signal
# by callers.
_PARSE_ERROR: List[str] = ["__unsafe_env_split_string__"]


def drop_option_arg(argv: List[str], index: int, opts_with_arg: frozenset[str]) -> int:
    """Return the index after skipping an option's argument, if any."""
    option = argv[index]
    index += 1
    if "=" in option:
        return index
    if option in opts_with_arg and index < len(argv):
        return index + 1
    return index


def _option_name(option: str) -> str:
    return option.split("=", 1)[0]


def strip_command_prefixes(argv: List[str]) -> List[str]:
    """Remove sudo/env/command/arch/nice/nohup and their options from *argv*.

    Returns a new list with the real command at position 0.
    ``env -S <cmd>`` is recursively split and stripped.
    On parse failure returns ``["__unsafe_env_split_string__"]`` so
    callers can fail-closed.
    """
    out = list(argv)
    while out:
        exe = os.path.basename(out[0]).lower()

        if exe == "sudo":
            out = out[1:]
            while out and out[0].startswith("-"):
                out = out[drop_option_arg(out, 0, SUDO_OPTIONS_WITH_ARG):]
            continue

        if exe == "env":
            out = out[1:]
            while out:
                if out[0] == "--":
                    out = out[1:]
                    continue

                opt_name = _option_name(out[0])

                if opt_name in _ENV_SPLIT_STRING_OPTIONS:
                    # env -S "tccutil reset …" → recursively parse
                    if "=" in out[0]:
                        split_string = out[0].split("=", 1)[1]
                    elif len(out) >= 2:
                        split_string = out[1]
                    else:
                        return _PARSE_ERROR
                    try:
                        return strip_command_prefixes(shlex.split(split_string))
                    except ValueError:
                        return _PARSE_ERROR

                if out[0].startswith("-"):
                    out = out[drop_option_arg(out, 0, ENV_OPTIONS_WITH_ARG):]
                    continue

                if "=" in out[0] and not out[0].startswith("-"):
                    out = out[1:]
                    continue
                break
            continue

        if exe == "command":
            out = out[1:]
            while out and out[0].startswith("-"):
                out = out[1:]
            continue

        if exe == "arch":
            out = out[1:]
            while out and out[0].startswith("-"):
                out = out[1:]
            continue

        if exe in {"nice", "nohup"}:
            out = out[1:]
            while out and out[0].startswith("-"):
                flag = out[0]
                out = out[1:]
                if flag == "-n" and out:
                    out = out[1:]
            continue

        break

    return out
