"""Shared types for the core tool-call policy system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol


class PolicySeverity:
    """Severity levels for policy decisions.

    ``CORE_BLOCK`` decisions are enforced by the core engine with fail-closed
    semantics.  ``USER_POLICY`` and ``ADVISORY`` decisions are delivered
    through the ordinary plugin hook system.
    """

    CORE_BLOCK = "core_block"
    USER_POLICY = "user_policy"
    ADVISORY = "advisory"

    _ALL = frozenset({CORE_BLOCK, USER_POLICY, ADVISORY})


@dataclass(frozen=True)
class PolicyDecision:
    """A single decision produced by a policy provider."""

    action: str  # "allow" | "block"
    error_type: str
    message: str
    provider: str
    severity: str = PolicySeverity.CORE_BLOCK


@dataclass(frozen=True)
class PolicyContext:
    """Read-only context supplied to every policy check."""

    tool_name: str
    tool_args: Mapping[str, Any]
    session_id: str = ""
    turn_id: str = ""
    platform: str = ""


class ToolPolicyProvider(Protocol):
    """A provider that inspects tool calls and returns a decision.

    Core providers MUST set ``fail_closed = True`` so that an unhandled
    exception during :meth:`check_tool_call` results in an automatic
    ``CORE_BLOCK`` decision rather than a silent pass-through.

    User-policy providers should set ``fail_closed = False`` and are run
    through the ordinary ``pre_tool_call`` plugin hook.
    """

    name: str
    fail_closed: bool

    def check_tool_call(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        context: PolicyContext,
    ) -> Optional[PolicyDecision]:
        """Inspect a tool call and return a decision, or ``None`` to allow."""
        ...
