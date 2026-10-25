"""Run-budget configuration normalization."""


def normalize_run_budget_seconds(value) -> float | None:
    """Positive float or None; bool must not turn YAML true into a one-second budget."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None
