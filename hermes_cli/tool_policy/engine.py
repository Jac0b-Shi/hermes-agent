"""Core policy engine — fail-closed enforcement of system-integrity rules.

This module is called from ``model_tools.handle_function_call()`` **before**
ordinary ``pre_tool_call`` plugin hooks.  It is NOT a plugin and cannot be
disabled or unloaded.

A provider crash here will block the tool call when the provider's
:attr:`~ToolPolicyProvider.fail_closed` flag is ``True`` — which is the
norm for core providers.  Provider **load** failures are also fail-closed:
if a core provider cannot be imported or constructed, every ``terminal``
tool call is blocked with a diagnostic until the underlying import error
is resolved.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .types import PolicyContext, PolicyDecision, PolicySeverity, ToolPolicyProvider

logger = logging.getLogger(__name__)

# Registered core providers — loaded lazily to avoid circular imports.
_CORE_PROVIDERS: Optional[List[ToolPolicyProvider]] = None
_CORE_LOAD_ERROR: Optional[str] = None


class CorePolicyLoadError(RuntimeError):
    """Raised when a core policy provider fails to load."""


def _load_core_providers() -> List[ToolPolicyProvider]:
    """Return the (cached) list of core policy providers.

    Raises :exc:`CorePolicyLoadError` if any core provider fails to
    import or construct — core providers are essential and a missing
    one means the integrity boundary is broken.
    """
    global _CORE_PROVIDERS, _CORE_LOAD_ERROR
    if _CORE_PROVIDERS is not None:
        return _CORE_PROVIDERS
    if _CORE_LOAD_ERROR is not None:
        raise CorePolicyLoadError(_CORE_LOAD_ERROR)

    providers: List[ToolPolicyProvider] = []
    try:
        from .providers.macos_system_integrity import MacOSSystemIntegrityProvider

        providers.append(MacOSSystemIntegrityProvider())
    except Exception as exc:
        _msg = f"failed to load macos-system-integrity provider: {exc}"
        logger.exception(_msg)
        _CORE_LOAD_ERROR = _msg
        raise CorePolicyLoadError(_msg) from exc

    _CORE_PROVIDERS = providers
    return providers


def enforce_core_tool_policy(
    tool_name: str,
    args: Dict[str, Any],
    *,
    session_id: str = "",
    turn_id: str = "",
) -> Optional[str]:
    """Run every registered core provider against a tool call.

    Returns a JSON block payload (same shape as the existing denylist /
    action-safety blocks) when a provider returns a ``CORE_BLOCK``
    decision, or ``None`` when every provider allows the call.

    Provider runtime exceptions are handled by the engine: if
    ``provider.fail_closed`` is ``True`` (the default for core providers),
    the tool call is blocked with a diagnostic message.
    """
    try:
        providers = _load_core_providers()
    except CorePolicyLoadError as exc:
        # Provider load failure → fail-closed for terminal commands
        if tool_name == "terminal":
            return _format_block(decision=PolicyDecision(
                action="block",
                error_type="core_policy_load_error",
                message=f"核心安全检查引擎加载失败，已按 fail-closed 拒绝执行: {exc}",
                provider="core_engine",
                severity=PolicySeverity.CORE_BLOCK,
            ))
        return None

    context = PolicyContext(
        tool_name=tool_name,
        tool_args=args,
        session_id=session_id,
        turn_id=turn_id,
    )

    for provider in providers:
        try:
            decision = provider.check_tool_call(tool_name, args, context)
        except Exception as exc:
            if provider.fail_closed:
                _msg = (
                    f"Blocked because core policy provider "
                    f"{provider.name!r} failed while checking {tool_name!r}: {exc}"
                )
                logger.exception("core policy provider %s failed", provider.name)
                return _format_block(decision=PolicyDecision(
                    action="block",
                    error_type="policy_provider_error",
                    message=_msg,
                    provider=provider.name,
                    severity=PolicySeverity.CORE_BLOCK,
                ))
            logger.warning("core policy provider %s failed (fail-open): %s", provider.name, exc)
            continue

        if decision is not None and decision.action == "block":
            logger.info(
                "core policy blocked tool=%s provider=%s error_type=%s",
                tool_name,
                provider.name,
                decision.error_type,
            )
            return _format_block(decision=decision)

    return None


def _format_block(decision: PolicyDecision) -> str:
    return json.dumps(
        {
            "status": "command_denied",
            "error_type": decision.error_type,
            "message": decision.message,
            "provider": decision.provider,
            "severity": decision.severity,
        },
        ensure_ascii=False,
    )
