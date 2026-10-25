"""Pure-data constructors shared by configuration defaults."""


def _aux(timeout, *, reasoning_effort=True, **extra):
    """Standard auxiliary-task model block; extra keys extend the independent route."""
    d = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        d["reasoning_effort"] = ""
    d.update(extra)
    return d
