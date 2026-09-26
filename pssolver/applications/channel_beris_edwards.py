"""P8.3 production application for complete-stress rectangular Channels."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pssolver.configuration.channel_beris_edwards import (
    ChannelBerisEdwardsRunSpec,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
)
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.workflows.channel_beris_edwards import (
    ChannelBerisEdwardsWorkflow,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_initial_q(run_spec: ChannelBerisEdwardsRunSpec):
    spec = run_spec.simulation
    snapshot = spec.initial_condition.parameters
    path = Path(snapshot["directory"]).expanduser().resolve() / (
        f"Q_{snapshot['step']}.npy"
    )
    values = np.load(path, allow_pickle=False)
    shape = tuple(spec.geometry.domain.shape)
    if values.shape == (*shape, len(Q_COMPONENTS)):
        values = np.moveaxis(values, -1, 0)
    elif values.shape != (len(Q_COMPONENTS), *shape):
        raise ValueError(
            "Channel Q snapshot must have shape (*grid, 5) or (5, *grid)"
        )
    expected_dtype = spec.numerics.precision.value
    if str(values.dtype) != expected_dtype:
        raise ValueError(
            f"Channel Q snapshot dtype must be {expected_dtype}, got {values.dtype}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Channel Q snapshot contains NaN or Inf")
    tensors = torch.from_numpy(np.array(values, copy=True))
    return (
        {name: tensors[index] for index, name in enumerate(Q_COMPONENTS)},
        path,
        _sha256(path),
    )


def run_channel_beris_edwards(
    run_spec: ChannelBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
):
    """Execute one frozen P8.3 application with no runtime fallback."""

    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be ChannelBerisEdwardsRunSpec")
    spec = run_spec.simulation
    metadata = {
        "schema_version": 1,
        "application": "channel_complete_stress_beris_edwards",
        "configuration": spec.to_metadata(),
        "configuration_sha256": spec.canonical_sha256(),
        "runtime_identity_sha256": run_spec.runtime_identity_sha256(),
        "runtime_selection": {
            "requested": run_spec.runtime_path,
            "effective": run_spec.runtime_path,
            "fallback_allowed": False,
            "fallback_used": False,
        },
        "pressure_gauge": "zero_mean",
        "velocity_nullspace": "absent_due_to_two_bounded_no_slip_axes",
    }
    if emit_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    if run_spec.dry_run:
        return None

    initial_values, snapshot_path, snapshot_sha256 = _load_initial_q(run_spec)
    metadata["initial_condition"] = {
        "source": str(snapshot_path),
        "sha256": snapshot_sha256,
        "logical_mode": spec.initial_condition.parameters["mode"],
    }
    output = run_spec.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty Channel output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    request = ChannelBerisEdwardsRuntimeBuildRequest(
        run_spec=run_spec,
        initial_values=initial_values,
        device=spec.execution.options["device"],
    )
    construction = PackageRuntimeConstructionInput(
        plan=plan_package_runtime_construction(spec),
        request=request,
    )
    adapter = build_package_simulation_runtime(construction)
    metadata["runtime_construction"] = construction.to_metadata()
    metadata["runtime"] = adapter.to_metadata()
    workflow = ChannelBerisEdwardsWorkflow(
        adapter,
        run_spec,
        output,
        metadata,
    )
    return workflow.run(progress=progress)


__all__ = ["run_channel_beris_edwards"]
