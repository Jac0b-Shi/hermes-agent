"""Tests for the core tool-call policy engine."""
from __future__ import annotations

import json
from unittest.mock import patch

from hermes_cli.tool_policy.engine import enforce_core_tool_policy
from hermes_cli.tool_policy.types import PolicyContext, PolicyDecision, PolicySeverity


def test_engine_allows_safe_commands():
    """Safe commands pass through without block."""
    assert enforce_core_tool_policy("terminal", {"command": "git status"}) is None
    assert enforce_core_tool_policy("terminal", {"command": "echo hello"}) is None


def test_engine_skips_non_terminal_tools():
    """Only terminal commands are inspected by the engine."""
    assert enforce_core_tool_policy("write_file", {"path": "/tmp/test", "content": "x"}) is None
    assert enforce_core_tool_policy("read_file", {"path": "/tmp/test"}) is None


def test_engine_load_failure_blocks_terminal():
    """When a core provider fails to load, terminal is fail-closed."""
    with patch(
        "hermes_cli.tool_policy.engine._CORE_PROVIDERS", None
    ), patch(
        "hermes_cli.tool_policy.engine._CORE_LOAD_ERROR", "mock import error"
    ):
        block = enforce_core_tool_policy("terminal", {"command": "git status"})
        assert block is not None
        payload = json.loads(block)
        assert payload["error_type"] == "core_policy_load_error"


def test_engine_load_failure_allows_non_terminal():
    """Load failure only blocks terminal, not other tools."""
    with patch(
        "hermes_cli.tool_policy.engine._CORE_PROVIDERS", None
    ), patch(
        "hermes_cli.tool_policy.engine._CORE_LOAD_ERROR", "mock import error"
    ):
        assert enforce_core_tool_policy("write_file", {"path": "/tmp/test", "content": "x"}) is None


class FakeFailClosedProvider:
    name = "fake_fail_closed"
    fail_closed = True

    def check_tool_call(self, tool_name, args, context):
        raise RuntimeError("simulated provider crash")


class FakeFailOpenProvider:
    name = "fake_fail_open"
    fail_closed = False

    def check_tool_call(self, tool_name, args, context):
        raise RuntimeError("simulated provider crash")


class FakeBlockingProvider:
    name = "fake_blocker"
    fail_closed = True

    def check_tool_call(self, tool_name, args, context):
        return PolicyDecision(
            action="block",
            error_type="test_block",
            message="test",
            provider="fake_blocker",
            severity=PolicySeverity.CORE_BLOCK,
        )


class FakeAllowProvider:
    name = "fake_allow"
    fail_closed = True

    def check_tool_call(self, tool_name, args, context):
        return None


def test_provider_runtime_exception_fail_closed_blocks():
    """Fail-closed provider crash blocks the tool call."""
    with patch(
        "hermes_cli.tool_policy.engine._load_core_providers",
        return_value=[FakeFailClosedProvider()],
    ):
        block = enforce_core_tool_policy("terminal", {"command": "git status"})
        assert block is not None
        payload = json.loads(block)
        assert payload["error_type"] == "policy_provider_error"


def test_provider_runtime_exception_fail_open_allows():
    """Fail-open provider crash does NOT block the tool call."""
    with patch(
        "hermes_cli.tool_policy.engine._load_core_providers",
        return_value=[FakeFailOpenProvider()],
    ):
        assert enforce_core_tool_policy("terminal", {"command": "git status"}) is None


def test_provider_blocks_returns_payload():
    """A blocking provider returns a JSON payload."""
    with patch(
        "hermes_cli.tool_policy.engine._load_core_providers",
        return_value=[FakeBlockingProvider()],
    ):
        block = enforce_core_tool_policy("terminal", {"command": "git status"})
        payload = json.loads(block)
        assert payload["status"] == "command_denied"
        assert payload["error_type"] == "test_block"
        assert payload["provider"] == "fake_blocker"


def test_provider_allow_returns_none():
    """An allowing provider returns None."""
    with patch(
        "hermes_cli.tool_policy.engine._load_core_providers",
        return_value=[FakeAllowProvider()],
    ):
        assert enforce_core_tool_policy("terminal", {"command": "git status"}) is None
