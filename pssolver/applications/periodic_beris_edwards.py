"""P8.2 production application edge for a complete-stress periodic box."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.periodic_beris_edwards import (
    PeriodicBerisEdwardsRunSpec,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
)
from pssolver.workflows.periodic_beris_edwards import (
    PeriodicBerisEdwardsWorkflow,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_initial_q(run_spec: PeriodicBerisEdwardsRunSpec):
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
            "periodic Q snapshot must have shape (*grid, 5) or (5, *grid)"
        )
    expected_dtype = spec.numerics.precision.value
    if str(values.dtype) != expected_dtype:
        raise ValueError(
            f"periodic Q snapshot dtype must be {expected_dtype}, got {values.dtype}"
        )
    if not np.isfinite(values).all():
        raise ValueError("periodic Q snapshot contains NaN or Inf")
    tensors = torch.from_numpy(np.array(values, copy=True))
    return (
        {name: tensors[index] for index, name in enumerate(Q_COMPONENTS)},
        path,
        _sha256(path),
    )


def run_periodic_beris_edwards(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
):
    """Execute one frozen P8.2 application with no runtime fallback."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    metadata = {
        "schema_version": 1,
        "application": "periodic_complete_stress_beris_edwards",
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
        "uniform_velocity_mode_action": (
            "remove_all_uniform_velocity_and_force"
        ),
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
        raise FileExistsError(f"refusing nonempty periodic output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    request = PeriodicRuntimeBuildRequest(
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
    workflow = PeriodicBerisEdwardsWorkflow(
        adapter,
        run_spec,
        output,
        metadata,
    )
    return workflow.run(progress=progress)


__all__ = ["run_periodic_beris_edwards"]
