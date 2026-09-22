"""Core tool-call policy engine — fail-closed enforcement boundary.

The engine runs as the last gate inside ``model_tools._execute_tool._dispatch()``
immediately before ``registry.dispatch()``, so it sees the final canonical args
after ``tool_request`` middleware, ``pre_tool_call`` modify/block, and
``tool_execution`` middleware rewrites.  skip_* flags never bypass it.  It is
**not** a plugin: it cannot be disabled, cannot be unloaded, and a provider
crash will block the tool call when
:attr:`ToolPolicyProvider.fail_closed` is ``True``.

Plugins are for user-preference policies (rm→trash, no git push --force).
The core engine is for system-integrity invariants (TCC reset, SIP
tampering, profiles mutation, …) that must never execute regardless
of plugin state or user configuration.
"""
