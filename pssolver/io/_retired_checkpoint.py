"""Early, tensor-free rejection of the three v0.5 retired restart families."""

from collections.abc import Mapping

from .checkpoint import CheckpointCompatibilityError


def reject_retired_checkpoint(metadata: Mapping[str, object], *, operation: str) -> None:
    """Reject before resolving or reading payload files; never migrate implicitly."""
    runtime = metadata.get("runtime_path")
    family = None
    if runtime in ("legacy_channel", "compiled_channel_v2"):
        family = "legacy active-force Channel"
    elif runtime == "separated_canary":
        family = "separated_canary"
    elif runtime == "legacy_production" and metadata.get("lifting_restart") is not None:
        family = "private Plane lifting"
    if family is not None:
        raise CheckpointCompatibilityError(
            f"{family} checkpoint is retired in v0.5; use v0.4.x (maint/v0.4) to resume",
            operation=operation,
        )
