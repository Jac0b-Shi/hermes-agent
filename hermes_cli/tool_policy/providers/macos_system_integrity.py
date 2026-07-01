"""Core provider: macOS system-integrity invariants.

These rules must never execute regardless of Hermes safety mode or
plugin state.  They protect TCC privacy decisions, SIP, Gatekeeper,
system extensions, configuration profiles, and the system-settings UI.

The provider activates **only** on local macOS terminal backends.
Remote, Docker, and non-macOS environments are intentionally
unaffected so that these rules do not produce false positives in CI,
SSH sessions, or container workloads.

This is **not** a plugin — it runs in the core engine's fail-closed
boundary and cannot be disabled.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..types import PolicyContext, PolicyDecision

logger = logging.getLogger(__name__)

# ── Deny lists ─────────────────────────────────────────────────────────

_DENY_TCC_SERVICES: frozenset[str] = frozenset({
    "all", "accessibility", "listenevent", "postevent", "appleevents",
    "screencapture", "systempolicyallfiles", "systempolicydesktopfolder",
    "systempolicydocumentsfolder", "systempolicydownloadsfolder",
    "systempolicynetworkvolumes", "systempolicyremovablevolumes",
    "camera", "microphone", "location", "photos", "addressbook",
    "calendar", "reminders", "bluetoothalways", "speechrecognition",
    "developertool", "remotedesktop",
})

_TCC_DB_MARKERS: Tuple[str, ...] = (
    "com.apple.tcc/tcc.db",
    "application support/com.apple.tcc/tcc.db",
)

_SHELL_NAMES: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash"})

_OSASCRIPT_RE = re.compile(r"(?:^|/)osascript$", re.IGNORECASE)
_SYSTEMPREFS_URL_RE = re.compile(r"x-apple\.systempreferences:", re.IGNORECASE)

# ── Local macOS gate ───────────────────────────────────────────────────


def _is_local_macos_terminal() -> bool:
    """Return True when the terminal tool is running on local macOS.

    Docker, SSH, Daytone, and remote backends are excluded.
    """
    if sys.platform != "darwin":
        return False
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    # FUTURE: HERMES_TERMINAL_BACKEND is not yet a canonical Hermes env var.
    # Short-term, the primary guard is TERMINAL_ENV.  The backend check is
    # best-effort; SSH/Docker remote backends should reliably set
    # TERMINAL_ENV != "local" so the gate holds even without this variable.
    # Medium-term, terminal tool should pass backend_kind / is_local /
    # host_platform explicitly via function_args or PolicyContext.
    terminal_backend = os.getenv("HERMES_TERMINAL_BACKEND", "")
    return terminal_env == "local" and terminal_backend not in {
        "docker", "ssh", "modal", "daytona", "singularity",
    }

# ── Shared shell helpers ───────────────────────────────────────────────
from ..shell_words import strip_command_prefixes as _strip_cmd_prefixes


def _find_shell_c_arg(argv: List[str]) -> Optional[str]:
    i = 0
    while i < len(argv):
        if argv[i] in ("-c", "-lc"):
            return argv[i + 1] if i + 1 < len(argv) else ""
        if argv[i].startswith("--") and i + 1 < len(argv) and argv[i + 1] == "-c":
            return argv[i + 2] if i + 2 < len(argv) else ""
        i += 1
    return None


def _check_command(raw_cmd: str) -> Optional[PolicyDecision]:
    """Inspect a raw shell command string for system-integrity violations."""
    if not raw_cmd.strip():
        return None

    try:
        argv = shlex.split(raw_cmd)
    except ValueError:
        return PolicyDecision(
            action="block",
            error_type="command_parse_error",
            message="无法安全解析 shell 命令，拒绝执行",
            provider="macos_system_integrity",
        )
    return _check_argv(raw_cmd, argv)


def _check_argv(raw_cmd: str, argv: List[str]) -> Optional[PolicyDecision]:
    exe_parts = _strip_cmd_prefixes(argv)
    if not exe_parts:
        return None

    # Sentinel from env -S parse failure → block
    if exe_parts == ["__unsafe_env_split_string__"]:
        return PolicyDecision(
            action="block",
            error_type="command_parse_error",
            message="无法安全解析 env -S 命令，拒绝执行",
            provider="macos_system_integrity",
        )

    exe_basename = os.path.basename(exe_parts[0]).lower()

    # Unwrap sh/bash/zsh/dash -c wrappers and recurse
    if exe_basename in _SHELL_NAMES:
        inner = _find_shell_c_arg(exe_parts)
        if inner is not None:
            try:
                return _check_command(inner)
            except Exception:
                return PolicyDecision(
                    action="block",
                    error_type="command_parse_error",
                    message="无法安全解析内嵌 shell 命令，拒绝执行",
                    provider="macos_system_integrity",
                )
        return None

    lowered_cmd = raw_cmd.lower()

    # 1. tccutil reset (any form, any service)
    if exe_basename == "tccutil":
        args_lower = [a.lower() for a in exe_parts[1:]]
        if len(args_lower) >= 1 and args_lower[0] == "reset":
            return PolicyDecision(
                action="block",
                error_type="dangerous_tcc_reset",
                message=(
                    "禁止 Hermes 执行 TCC 权限重置（高风险操作）。"
                    "这会清空 macOS 隐私授权状态，导致大量 App 权限丢失。"
                ),
                provider="macos_system_integrity",
            )

    # 2. TCC.db direct access
    if any(marker in lowered_cmd for marker in _TCC_DB_MARKERS):
        return PolicyDecision(
            action="block",
            error_type="dangerous_tcc_db_access",
            message=(
                "禁止 Hermes 访问 TCC 隐私数据库。"
                "修改 TCC.db 属于绕过系统设置界面，会破坏 macOS 权限系统。"
            ),
            provider="macos_system_integrity",
        )

    # 3. Configuration profile tampering
    if exe_basename == "profiles":
        args_lower = [a.lower() for a in exe_parts[1:]]
        if any(a in args_lower for a in ("install", "remove", "renew")):
            return PolicyDecision(
                action="block",
                error_type="dangerous_profiles_tampering",
                message=(
                    "禁止 Hermes 修改配置描述文件（profiles）。"
                    "profiles 可以安装/删除系统级配置，可能绕过安全策略。"
                ),
                provider="macos_system_integrity",
            )

    # 4. AppleScript
    if exe_basename == "osascript" or _OSASCRIPT_RE.search(exe_parts[0]):
        return PolicyDecision(
            action="block",
            error_type="dangerous_applescript",
            message=(
                "禁止 Hermes 执行 osascript / AppleScript。"
                "AppleScript 可配合 System Events 进行 UI 自动化，"
                "可能绕过 TCC 授权流程。"
            ),
            provider="macos_system_integrity",
        )

    # 5. Open system preferences
    if exe_basename == "open" and _SYSTEMPREFS_URL_RE.search(lowered_cmd):
        return PolicyDecision(
            action="block",
            error_type="dangerous_systemprefs_open",
            message=(
                "禁止 Hermes 打开系统设置面板。"
                "这可能在辅助功能权限下诱导或自动化隐私设置修改。"
            ),
            provider="macos_system_integrity",
        )

    # 6. SIP tampering
    if exe_basename == "csrutil":
        return PolicyDecision(
            action="block",
            error_type="dangerous_sip_tampering",
            message="禁止 Hermes 执行 csrutil（SIP 配置工具）。",
            provider="macos_system_integrity",
        )

    # 7. Gatekeeper disable
    if exe_basename == "spctl":
        args_lower = [a.lower() for a in exe_parts[1:]]
        if "--master-disable" in args_lower:
            return PolicyDecision(
                action="block",
                error_type="dangerous_gatekeeper_disable",
                message=(
                    "禁止 Hermes 执行 spctl --master-disable。"
                    "这会禁用 macOS Gatekeeper，允许任意未签名代码运行。"
                ),
                provider="macos_system_integrity",
            )

    # 8. System extension tampering
    if exe_basename == "systemextensionsctl":
        args_lower = [a.lower() for a in exe_parts[1:]]
        if any(a in args_lower for a in ("uninstall", "reset")):
            return PolicyDecision(
                action="block",
                error_type="dangerous_systemextensions_tampering",
                message=(
                    "禁止 Hermes 卸载/重置系统扩展。"
                    "系统扩展运行在内核空间，卸载可能导致系统不稳定。"
                ),
                provider="macos_system_integrity",
            )

    return None


# ── Provider ────────────────────────────────────────────────────────────


class MacOSSystemIntegrityProvider:
    """Core provider for macOS system-integrity invariants.

    Fail-closed: if this provider crashes, the tool call is blocked.
    Only activates on local macOS terminal backends.
    """

    name: str = "macos_system_integrity"
    fail_closed: bool = True

    def check_tool_call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        context: PolicyContext,
    ) -> Optional[PolicyDecision]:
        if (tool_name or "").lower() != "terminal":
            return None
        if not _is_local_macos_terminal():
            return None

        cmd = str(args.get("command") or args.get("cmd") or "")
        return _check_command(cmd)
