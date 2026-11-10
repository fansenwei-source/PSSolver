"""Step-local derivative reuse for future model terms (unused by Periodic)."""


class StepDerivativeTable(dict):
    """A caller-local table that never survives its execution step."""
