"""Tests for the macOS system-integrity core policy provider."""
from __future__ import annotations

import os
from unittest.mock import patch

from hermes_cli.tool_policy.providers.macos_system_integrity import (
    MacOSSystemIntegrityProvider,
    _check_command,
)

from hermes_cli.tool_policy.types import PolicyContext


def _ctx(tool_name="terminal", **kwargs):
    return PolicyContext(tool_name=tool_name, tool_args=kwargs)


# ── Direct command checks (no platform gate) ───────────────────────────


def _blocked(cmd):
    result = _check_command(cmd)
    if result is None:
        return None
    return result.error_type


def test_tccutil_reset_blocked():
    assert _blocked("tccutil reset SystemPolicyAllFiles") == "dangerous_tcc_reset"
    assert _blocked("tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("sudo tccutil reset Camera") == "dangerous_tcc_reset"


def test_tccutil_status_not_blocked():
    assert _blocked("tccutil status") is None
    assert _blocked("tccutil status SystemPolicyAllFiles") is None
    assert _blocked("tccutil list") is None


def test_tcc_db_access_blocked():
    assert _blocked("cat ~/Library/Application Support/com.apple.TCC/TCC.db") is not None
    assert _blocked("sqlite3 /Library/Application Support/com.apple.TCC/TCC.db") is not None


def test_profiles_blocked():
    assert _blocked("profiles install -path /tmp/test") == "dangerous_profiles_tampering"
    assert _blocked("profiles remove -all") == "dangerous_profiles_tampering"
    assert _blocked("profiles renew -type enrollment") == "dangerous_profiles_tampering"


def test_profiles_read_only_not_blocked():
    assert _blocked("profiles list") is None
    assert _blocked("profiles status") is None


def test_osascript_blocked():
    assert _blocked("osascript -e 'test'") == "dangerous_applescript"
    assert _blocked("/usr/bin/osascript script.scpt") == "dangerous_applescript"


def test_systemprefs_open_blocked():
    assert _blocked("open x-apple.systempreferences:com.apple.preference.security") == "dangerous_systemprefs_open"


def test_normal_open_not_blocked():
    assert _blocked("open https://example.com") is None


def test_csrutil_blocked():
    assert _blocked("csrutil disable") == "dangerous_sip_tampering"
    assert _blocked("sudo csrutil clear") == "dangerous_sip_tampering"


def test_spctl_blocked():
    assert _blocked("spctl --master-disable") == "dangerous_gatekeeper_disable"


def test_spctl_assess_not_blocked():
    assert _blocked("spctl --assess /path/to/app") is None


def test_systemextensionsctl_blocked():
    assert _blocked("systemextensionsctl uninstall TEAMID com.example") == "dangerous_systemextensions_tampering"


def test_systemextensionsctl_list_not_blocked():
    assert _blocked("systemextensionsctl list") is None


def test_safe_commands_not_blocked():
    for cmd in ["git status", "echo hello", "npm install", "ls -la"]:
        assert _blocked(cmd) is None


# ── Shell wrapper ──────────────────────────────────────────────────────


def test_shell_wrapper_inner_dangerous_blocked():
    assert _blocked("sh -c 'tccutil reset Accessibility'") == "dangerous_tcc_reset"
    assert _blocked("zsh -c 'csrutil disable'") == "dangerous_sip_tampering"


def test_shell_wrapper_inner_safe_not_blocked():
    assert _blocked("sh -c 'echo hello'") is None


# ── Prefix parser bypass tests ─────────────────────────────────────────


def test_sudo_option_bypass_not_possible():
    """sudo -g/-p/-h/etc. options should not hide the dangerous command."""
    assert _blocked("env -u FOO tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("sudo -g staff tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("sudo -p prompt tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("sudo -h hostname tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("env --unset FOO tccutil reset Accessibility") == "dangerous_tcc_reset"
    # env with interleaved options and assignments
    assert _blocked("env -u FOO BAR=baz tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("env -i BAR=baz tccutil reset Accessibility") == "dangerous_tcc_reset"
    assert _blocked("env --unset FOO BAR=baz tccutil reset Accessibility") == "dangerous_tcc_reset"


# ── env -S / --split-string bypass tests ───────────────────────────────


def test_env_split_string_variants_blocked():
    assert _blocked('env -S "tccutil reset Accessibility"') == "dangerous_tcc_reset"
    assert _blocked('env --split-string "tccutil reset Accessibility"') == "dangerous_tcc_reset"
    assert _blocked('env --split-string="tccutil reset Accessibility"') == "dangerous_tcc_reset"
    assert _blocked('env -S "sudo tccutil reset Camera"') == "dangerous_tcc_reset"


def test_env_split_string_safe_commands_not_blocked():
    assert _blocked('env -S "echo hello"') is None
    assert _blocked('env -S "git status"') is None


def test_env_split_string_malformed_blocks():
    """Malformed env -S (unterminated quote) → fail-closed block."""
    assert _blocked('env -S "unterminated') == "command_parse_error"


# ── Provider gate tests ────────────────────────────────────────────────

_provider = MacOSSystemIntegrityProvider()


def _provider_blocked(command, tool_name="terminal", **env_overrides):
    """Run provider check with env overrides. Returns error_type or None."""
    with patch.dict(os.environ, {k: v for k, v in env_overrides.items() if v is not None}, clear=False):
        decision = _provider.check_tool_call(
            tool_name, {"command": command}, _ctx(tool_name, command=command)
        )
    return decision.error_type if decision else None


def test_provider_non_terminal_returns_none():
    assert _provider.check_tool_call("write_file", {"path": "/tmp/test", "content": "x"}, _ctx("write_file")) is None


def test_provider_non_darwin_returns_none():
    with patch("sys.platform", "win32"):
        assert _provider.check_tool_call("terminal", {"command": "tccutil reset All"}, _ctx()) is None


def test_provider_local_darwin_blocks():
    with patch("sys.platform", "darwin"), patch.dict(os.environ, {"TERMINAL_ENV": "local"}):
        decision = _provider.check_tool_call("terminal", {"command": "tccutil reset All"}, _ctx())
        assert decision is not None
        assert decision.error_type == "dangerous_tcc_reset"


def test_provider_docker_backend_bypasses():
    with patch("sys.platform", "darwin"), patch.dict(os.environ, {"TERMINAL_ENV": "docker"}, clear=True):
        assert _provider.check_tool_call("terminal", {"command": "tccutil reset All"}, _ctx()) is None


def test_provider_ssh_backend_bypasses():
    with patch("sys.platform", "darwin"), patch.dict(os.environ, {"TERMINAL_ENV": "ssh"}, clear=True):
        assert _provider.check_tool_call("terminal", {"command": "osascript test"}, _ctx()) is None


def test_provider_backend_env_var_bypasses():
    """HERMES_TERMINAL_BACKEND=docker bypasses even with local TERMINAL_ENV."""
    with patch("sys.platform", "darwin"), patch.dict(
        os.environ,
        {"TERMINAL_ENV": "local", "HERMES_TERMINAL_BACKEND": "docker"},
        clear=True,
    ):
        assert _provider.check_tool_call("terminal", {"command": "tccutil reset All"}, _ctx()) is None
