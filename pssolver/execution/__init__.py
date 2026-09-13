"""Execution-facing contracts independent of the legacy solver objects."""

from .contracts import ExecutableModelProtocol, ModelExecutionContext

__all__ = [
    "ExecutableModelProtocol",
    "ModelExecutionContext",
]
