"""Core tool-call policy engine — fail-closed enforcement boundary.

The engine runs in the *preamble* of ``model_tools.handle_function_call()``,
before ordinary ``pre_tool_call`` plugin hooks, action-safety, and tool
dispatch.  It is **not** a plugin: it cannot be disabled, cannot be
unloaded, and a provider crash will block the tool call when
:attr:`ToolPolicyProvider.fail_closed` is ``True``.

Plugins are for user-preference policies (rm→trash, no git push --force).
The core engine is for system-integrity invariants (TCC reset, SIP
tampering, profiles mutation, …) that must never execute regardless
of plugin state or user configuration.
"""
