#!/usr/bin/env python3
"""Run the frozen RC4.2.6 CPU/CUDA checkpoint-portability matrix.

The runner is deliberately an external qualification client.  It uses the
installed package API, writes one fail-closed JSON record per matrix cell, and
never overwrites an existing output.  ``--plan-only`` is allocation-free and
is used by the login-node preflight to authenticate the exact X01--X14 order.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
from typing import Callable, Iterable


SCHEMA = "pssolver.rc4_2_6.cross_device_checkpoint_cell.v1"
PLAN_SCHEMA = "pssolver.rc4_2_6.cross_device_checkpoint_plan.v1"
COMPLETE_SCHEMA = "pssolver.rc4_2_6.cross_device_checkpoint_complete.v1"
PASS_CLASSIFICATION = (
    "PASS_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY"
)


@dataclass(frozen=True, slots=True)
class MatrixCell:
    sequence: int
    id: str
    family: str
    runtime_path: str
    source: str
    target: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "id": self.id,
            "family": self.family,
            "runtime_path": self.runtime_path,
            "source": self.source,
            "target": self.target,
        }


_ROWS = (
    ("plane_production", "legacy_production"),
    ("plane_production", "compiled_v2"),
    ("plane_production", "separated_canary"),
    ("periodic_production", "periodic_spectral"),
    ("channel_production", "channel_complete_stress"),
    ("periodic_functional", "periodic_activity_batch_one"),
    ("channel_functional", "channel_activity_batch_one"),
)
MATRIX = tuple(
    MatrixCell(
        sequence=index + 1,
        id=f"X{index + 1:02d}",
        family=family,
        runtime_path=runtime_path,
        source=source,
        target=target,
    )
    for index, (family, runtime_path, source, target) in enumerate(
        (
            (family, runtime_path, source, target)
            for family, runtime_path in _ROWS
            for source, target in (("cpu", "cuda"), ("cuda", "cpu"))
        )
    )
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"atomic JSON round-trip differs for {path}")
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _payload_sha256(directory: Path) -> dict[str, str]:
    return {
        name: value
        for name, value in _tree_sha256(directory).items()
        if name.endswith(".npy")
    }


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _compatibility_sha256(directory: Path) -> str:
    metadata = _json(directory / "checkpoint.json")
    direct = metadata.get("compatibility_identity_sha256")
    if isinstance(direct, str):
        return direct
    bridge = metadata.get("functional_bridge")
    if isinstance(bridge, dict):
        value = bridge.get("checkpoint_compatibility_sha256")
        if isinstance(value, str):
            return value
    raise ValueError("checkpoint compatibility SHA-256 is missing")


def _progress_metadata(directory: Path) -> dict[str, object]:
    metadata = _json(directory / "checkpoint.json")
    result: dict[str, object] = {
        "completed_steps": metadata.get("completed_steps")
    }
    integrator = metadata.get("integrator")
    if integrator is not None:
        if not isinstance(integrator, dict):
            raise TypeError("checkpoint integrator progress must be an object")
        result["integrator"] = integrator
    return result


def _ordinary_checkpoint(directory: Path) -> bool:
    return directory.is_dir() and not directory.is_symlink()


def _corrupt_copy(source: Path, target: Path) -> Path:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"tamper target already exists: {target}")
    shutil.copytree(source, target)
    payloads = sorted(target.rglob("*.npy"))
    if not payloads:
        raise RuntimeError("checkpoint has no tensor payload to tamper")
    path = payloads[0]
    values = bytearray(path.read_bytes())
    if not values:
        raise RuntimeError("checkpoint tensor payload is empty")
    values[-1] ^= 1
    path.write_bytes(bytes(values))
    return path


def _tensor_sha256(values: Iterable[object]) -> str:
    import torch

    digest = hashlib.sha256()
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("state fingerprint accepts only tensors")
        current = value.detach().contiguous().cpu()
        digest.update(str(current.dtype).encode("ascii"))
        digest.update(json.dumps(list(current.shape)).encode("ascii"))
        digest.update(current.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite(values: Iterable[object]) -> bool:
    import torch

    return all(
        isinstance(value, torch.Tensor)
        and bool(torch.isfinite(value).all().item())
        for value in values
    )


def _devices(values: Iterable[object]) -> list[str]:
    import torch

    return sorted(
        {
            str(value.device)
            for value in values
            if isinstance(value, torch.Tensor)
        }
    )


def _analytic_q(shape: tuple[int, int, int]):
    import numpy as np

    axes = np.indices(shape, dtype=np.float64)
    x = 2.0 * np.pi * axes[0] / shape[0]
    y = 2.0 * np.pi * axes[1] / shape[1]
    z = 2.0 * np.pi * axes[2] / shape[2]
    values = np.empty((*shape, 5), dtype=np.float64)
    values[..., 0] = 0.20 + 1.0e-3 * np.sin(x + y)
    values[..., 1] = 1.0e-3 * np.cos(y + z)
    values[..., 2] = 1.0e-3 * np.sin(x + z)
    values[..., 3] = -0.10 + 1.0e-3 * np.cos(x - y)
    values[..., 4] = 1.0e-3 * np.sin(y - z)
    return values


def _snapshot(directory: Path, shape: tuple[int, int, int]) -> Path:
    import numpy as np

    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"snapshot directory already exists: {directory}")
    directory.mkdir(parents=True)
    with (directory / "Q_0.npy").open("xb") as stream:
        np.save(stream, _analytic_q(shape), allow_pickle=False)
    return directory


def _model(*, periodic: bool = False):
    from pssolver.models.active_nematics import CompleteStressBerisEdwards

    options: dict[str, object] = {}
    if periodic:
        options.update(
            friction=0.0,
            tangential_zero_mode_policy="zero_mean",
        )
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
        **options,
    )


def _periodic_simulation(root: Path, label: str, device: str):
    from pssolver import (
        Output,
        Simulation,
        SnapshotInitialCondition,
        SpectralNumerics,
        TimeStepping,
        TorchSpectralExecution,
    )
    from pssolver.boundaries import (
        assign_boundaries,
        free_slip_velocity,
        neumann_pressure_compatibility,
        neumann_q,
    )
    from pssolver.geometries import PeriodicBox

    shape = (6, 6, 4)
    model = _model(periodic=True)
    geometry = PeriodicBox(shape=shape, lengths=(6.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            _snapshot(root / f"{label}_snapshot", shape),
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=root / f"{label}_output",
            steps=2,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def _channel_simulation(root: Path, label: str, device: str):
    from pssolver import (
        Output,
        Simulation,
        SnapshotInitialCondition,
        SpectralNumerics,
        TimeStepping,
        TorchSpectralExecution,
    )
    from pssolver.boundaries import (
        assign_boundaries,
        neumann_pressure_compatibility,
        neumann_q,
        no_slip_velocity,
    )
    from pssolver.geometries import RectangularChannel

    shape = (8, 6, 4)
    model = _model()
    geometry = RectangularChannel(shape=shape, lengths=(8.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            _snapshot(root / f"{label}_snapshot", shape),
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=root / f"{label}_output",
            steps=2,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )


def _plane_runtime(root: Path, label: str, device: str, runtime_path: str):
    import torch
    from pssolver.configuration import create_plane_beris_edwards_run_spec
    from pssolver.configuration.active_nematics_simulation_adapters import (
        compose_plane_beris_edwards_simulation,
    )
    from pssolver.configuration.package_construction import (
        plan_package_runtime_construction,
    )
    from pssolver.configuration.plane_beris_edwards_components import (
        decompose_plane_beris_edwards_run_spec,
    )
    from pssolver.models.active_nematics import Q_COMPONENTS
    from pssolver.applications.plane_beris_edwards import (
        run_plane_beris_edwards,
    )
    from pssolver.runtime.package_construction import (
        PackageRuntimeConstructionInput,
        build_package_simulation_runtime,
    )
    from pssolver.runtime.plane_beris_edwards import (
        PlaneRuntimeBuildRequest,
        build_plane_beris_edwards_runtime,
    )

    spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=root / f"{label}_output",
        device=device,
        dtype="float64",
        pointwise_execution="eager",
        runtime_path=runtime_path,
        nx=8,
        ny=8,
        nz=6,
        lx=8.0,
        ly=9.0,
        height=20.0,
        dt=0.001,
        steps=2,
        save_start_step=0,
        save_interval=2,
        diagnostic_interval=2,
        spectral_refresh_steps=None,
        seed=24,
        defect_min_separation=2.0,
    )
    values = _analytic_q((8, 8, 6))
    initial = {
        name: torch.from_numpy(values[..., index].copy())
        for index, name in enumerate(Q_COMPONENTS)
    }
    simulation = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(spec)
    )
    production_metadata = {
        "configuration": spec.identity_metadata(),
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    if runtime_path == "separated_canary":
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = run_plane_beris_edwards(
                replace(spec, dry_run=True),
                emit_metadata=True,
            )
        if result is not None:
            raise RuntimeError("Plane dry-run unexpectedly executed")
        production_metadata = json.loads(stream.getvalue())
        production_metadata["configuration"] = spec.identity_metadata()
        production_metadata["runtime_selection"] = (
            spec.runtime_selection_metadata()
        )
    request = PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata=production_metadata,
        initial_values=initial,
        device=device,
        application_simulation=simulation,
    )
    if runtime_path == "separated_canary":
        adapter = build_plane_beris_edwards_runtime(request)
    else:
        adapter = build_package_simulation_runtime(
            PackageRuntimeConstructionInput(
                plan_package_runtime_construction(simulation),
                request,
            )
        )
    return spec, adapter


def _periodic_production_runtime(root: Path, label: str, device: str):
    from pssolver import compile_simulation
    from pssolver.applications.periodic_beris_edwards import (
        load_periodic_initial_q,
    )
    from pssolver.runtime.periodic_beris_edwards import (
        PeriodicRuntimeBuildRequest,
        build_periodic_beris_edwards_runtime,
    )

    simulation = _periodic_simulation(root, label, device)
    spec = compile_simulation(simulation).application_request
    initial, _, _ = load_periodic_initial_q(spec)
    return spec, build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(spec, initial, device)
    )


def _channel_production_runtime(root: Path, label: str, device: str):
    from pssolver import compile_simulation
    from pssolver.applications.channel_beris_edwards import load_channel_initial_q
    from pssolver.runtime.channel_beris_edwards import (
        ChannelBerisEdwardsRuntimeBuildRequest,
        build_channel_beris_edwards_runtime,
    )

    simulation = _channel_simulation(root, label, device)
    spec = compile_simulation(simulation).application_request
    initial, _, _ = load_channel_initial_q(spec)
    return spec, build_channel_beris_edwards_runtime(
        ChannelBerisEdwardsRuntimeBuildRequest(spec, initial, device)
    )


def _functional_runtime(
    root: Path,
    label: str,
    device: str,
    *,
    channel: bool,
):
    from pssolver.functional.api import (
        build_channel_activity_functional_runtime,
        build_functional_runtime,
        channel_activity_functional_request,
        periodic_activity_functional_request,
    )

    simulation = (
        _channel_simulation(root, label, device)
        if channel
        else _periodic_simulation(root, label, device)
    )
    request = (
        channel_activity_functional_request(simulation.specification)
        if channel
        else periodic_activity_functional_request(simulation.specification)
    )
    runtime = (
        build_channel_activity_functional_runtime(request)
        if channel
        else build_functional_runtime(request)
    )
    return runtime


def _production_state(adapter, *, channel: bool = False) -> tuple[object, ...]:
    from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS

    names = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p") if channel else Q_COMPONENTS
    values = tuple(
        adapter.fields[f"{name}{suffix}"]
        for name in names
        for suffix in ("", ".hat")
    )
    if channel:
        values += (adapter.capture_pressure_guess(),)
    return values


def _runtime_fallback(adapter) -> bool:
    metadata = adapter.to_metadata()
    return bool(metadata.get("fallback_used", False))


def _checkpoint_common(
    cell: MatrixCell,
    root: Path,
    *,
    source_values: tuple[object, ...],
    target_values_before: tuple[object, ...],
    target_values_after: Callable[[], tuple[object, ...]],
    checkpoint: Path,
    target_probe: Path,
    restored_checkpoint: Path,
    tampered_checkpoint: Path,
    restore: Callable[[Path], int],
    post_restore_step: Callable[[], None],
    completed_steps: int,
    fallback_used: bool,
) -> dict[str, object]:
    source_tree_before = _tree_sha256(checkpoint)
    source_payload = _payload_sha256(checkpoint)
    source_compatibility = _compatibility_sha256(checkpoint)
    target_compatibility = _compatibility_sha256(target_probe)
    before_fingerprint = _tensor_sha256(target_values_before)
    tampered_file = _corrupt_copy(checkpoint, tampered_checkpoint)
    rejection: dict[str, object]
    try:
        restore(tampered_checkpoint)
    except Exception as exc:  # the exact integrity class is family-specific
        message = str(exc)
        rejection = {
            "rejected": True,
            "exception_type": type(exc).__name__,
            "message": message,
            "integrity_guard_reached": "checksum" in message.lower(),
            "tampered_file": tampered_file.relative_to(tampered_checkpoint).as_posix(),
        }
    else:
        rejection = {
            "rejected": False,
            "exception_type": None,
            "message": None,
            "integrity_guard_reached": False,
        }
    after_rejection = target_values_after()
    rejection["target_unchanged"] = (
        before_fingerprint == _tensor_sha256(after_rejection)
    )
    restored_steps = restore(checkpoint)
    restored_values = target_values_after()
    post_restore_payload_before_step = _tensor_sha256(restored_values)
    post_restore_step()
    post_step_values = target_values_after()
    source_progress = _progress_metadata(checkpoint)
    restored_progress = _progress_metadata(restored_checkpoint)
    result = {
        "schema": SCHEMA,
        "cell": cell.to_metadata(),
        "checkpoint": {
            "ordinary_non_symlink_directory": _ordinary_checkpoint(checkpoint),
            "source_tree_unchanged": source_tree_before == _tree_sha256(checkpoint),
            "source_tree_sha256": source_tree_before,
            "source_payload_sha256": source_payload,
            "restored_payload_sha256": _payload_sha256(restored_checkpoint),
            "compatibility_identity_source": source_compatibility,
            "compatibility_identity_target": target_compatibility,
            "compatibility_identity_equal": source_compatibility
            == target_compatibility,
            "negative_integrity_gate": rejection,
        },
        "progress": {
            "expected_completed_steps": completed_steps,
            "restored_completed_steps": restored_steps,
            "exact": restored_steps == completed_steps,
            "source_serialized": source_progress,
            "restored_serialized": restored_progress,
            "serialized_equal": source_progress == restored_progress,
        },
        "state": {
            "source_devices": _devices(source_values),
            "target_devices": _devices(restored_values),
            "serialized_payload_equal": source_payload
            == _payload_sha256(restored_checkpoint),
            "persistent_backend_state_equal": source_payload
            == _payload_sha256(restored_checkpoint),
            "restored_tensor_sha256": post_restore_payload_before_step,
            "restored_finite": _finite(restored_values),
            "post_restore_step_finite": _finite(post_step_values),
        },
        "runtime": {"fallback_used": fallback_used},
    }
    gates = (
        result["checkpoint"]["ordinary_non_symlink_directory"],
        result["checkpoint"]["source_tree_unchanged"],
        result["checkpoint"]["compatibility_identity_equal"],
        rejection["rejected"],
        rejection["integrity_guard_reached"],
        rejection["target_unchanged"],
        result["progress"]["exact"],
        result["progress"]["serialized_equal"],
        result["state"]["serialized_payload_equal"],
        result["state"]["persistent_backend_state_equal"],
        result["state"]["restored_finite"],
        result["state"]["post_restore_step_finite"],
        not fallback_used,
    )
    result["passed"] = all(bool(value) for value in gates)
    return result


def _run_plane(cell: MatrixCell, root: Path) -> dict[str, object]:
    from pssolver.workflows.plane_checkpoint import (
        capture_plane_checkpoint,
        load_plane_checkpoint,
        restore_plane_checkpoint,
        write_plane_checkpoint,
    )

    source_spec, source = _plane_runtime(
        root, "source", cell.source, cell.runtime_path
    )
    target_spec, target = _plane_runtime(
        root, "target", cell.target, cell.runtime_path
    )
    source.advance(1)
    checkpoint = write_plane_checkpoint(
        root / "checkpoint", capture_plane_checkpoint(source, run_spec=source_spec)
    )
    write_plane_checkpoint(
        root / "target_probe",
        capture_plane_checkpoint(target, run_spec=target_spec),
    )
    source_values = _production_state(source)
    target_before = _production_state(target)

    def restore(path: Path) -> int:
        return restore_plane_checkpoint(
            target,
            load_plane_checkpoint(path),
            run_spec=target_spec,
        )

    def post_step() -> None:
        write_plane_checkpoint(
            root / "restored_checkpoint",
            capture_plane_checkpoint(target, run_spec=target_spec),
        )
        target.advance(1)

    return _checkpoint_common(
        cell,
        root,
        source_values=source_values,
        target_values_before=target_before,
        target_values_after=lambda: _production_state(target),
        checkpoint=checkpoint,
        target_probe=root / "target_probe",
        restored_checkpoint=root / "restored_checkpoint",
        tampered_checkpoint=root / "tampered_checkpoint",
        restore=restore,
        post_restore_step=post_step,
        completed_steps=1,
        fallback_used=_runtime_fallback(source) or _runtime_fallback(target),
    )


def _run_periodic(cell: MatrixCell, root: Path) -> dict[str, object]:
    from pssolver.workflows.periodic_checkpoint import (
        capture_periodic_checkpoint,
        load_periodic_checkpoint,
        restore_periodic_checkpoint,
        write_periodic_checkpoint,
    )

    source_spec, source = _periodic_production_runtime(root, "source", cell.source)
    target_spec, target = _periodic_production_runtime(root, "target", cell.target)
    source.advance(1)
    checkpoint = write_periodic_checkpoint(
        root / "checkpoint",
        capture_periodic_checkpoint(source, run_spec=source_spec),
    )
    write_periodic_checkpoint(
        root / "target_probe",
        capture_periodic_checkpoint(target, run_spec=target_spec),
    )
    source_values = _production_state(source)
    target_before = _production_state(target)

    def restore(path: Path) -> int:
        return restore_periodic_checkpoint(
            target,
            load_periodic_checkpoint(path),
            run_spec=target_spec,
        )

    def post_step() -> None:
        write_periodic_checkpoint(
            root / "restored_checkpoint",
            capture_periodic_checkpoint(target, run_spec=target_spec),
        )
        target.advance(1)

    return _checkpoint_common(
        cell,
        root,
        source_values=source_values,
        target_values_before=target_before,
        target_values_after=lambda: _production_state(target),
        checkpoint=checkpoint,
        target_probe=root / "target_probe",
        restored_checkpoint=root / "restored_checkpoint",
        tampered_checkpoint=root / "tampered_checkpoint",
        restore=restore,
        post_restore_step=post_step,
        completed_steps=1,
        fallback_used=_runtime_fallback(source) or _runtime_fallback(target),
    )


def _run_channel(cell: MatrixCell, root: Path) -> dict[str, object]:
    from pssolver.workflows.channel_beris_edwards import (
        _load_checkpoint,
        write_channel_beris_edwards_checkpoint,
    )

    source_spec, source = _channel_production_runtime(root, "source", cell.source)
    target_spec, target = _channel_production_runtime(root, "target", cell.target)
    source.advance(1)
    checkpoint = root / "checkpoint"
    write_channel_beris_edwards_checkpoint(checkpoint, source, run_spec=source_spec)
    write_channel_beris_edwards_checkpoint(
        root / "target_probe", target, run_spec=target_spec
    )
    source_values = _production_state(source, channel=True)
    target_before = _production_state(target, channel=True)

    def restore(path: Path) -> int:
        return _load_checkpoint(path, target, run_spec=target_spec)

    def post_step() -> None:
        write_channel_beris_edwards_checkpoint(
            root / "restored_checkpoint", target, run_spec=target_spec
        )
        target.advance(1)

    return _checkpoint_common(
        cell,
        root,
        source_values=source_values,
        target_values_before=target_before,
        target_values_after=lambda: _production_state(target, channel=True),
        checkpoint=checkpoint,
        target_probe=root / "target_probe",
        restored_checkpoint=root / "restored_checkpoint",
        tampered_checkpoint=root / "tampered_checkpoint",
        restore=restore,
        post_restore_step=post_step,
        completed_steps=1,
        fallback_used=_runtime_fallback(source) or _runtime_fallback(target),
    )


def _run_functional(cell: MatrixCell, root: Path) -> dict[str, object]:
    channel = cell.family == "channel_functional"
    source = _functional_runtime(root, "source", cell.source, channel=channel)
    target = _functional_runtime(root, "target", cell.target, channel=channel)
    source_state = source.initial_state()
    source_control_spec = source.control_specs[0].tensor
    import torch

    source_controls = {
        "activity": torch.full(
            source_control_spec.shape,
            0.01,
            dtype=torch.float64,
            device=source_control_spec.device,
        )
    }
    source_state = source.step(source_state, source_controls, 0)
    source_bridge = source.checkpoint_bridge
    target_bridge = target.checkpoint_bridge
    _require(source_bridge is not None and target_bridge is not None, "bridge missing")
    checkpoint = source_bridge.export_checkpoint(
        root / "checkpoint", source_state, completed_steps=1
    )
    target_initial = target.initial_state()
    target_bridge.export_checkpoint(
        root / "target_probe", target_initial, completed_steps=0
    )
    imported_holder: dict[str, object] = {}

    def restore(path: Path) -> int:
        imported = target_bridge.import_checkpoint(path)
        imported_holder["state"] = imported.state
        return imported.completed_steps

    def values_after() -> tuple[object, ...]:
        value = imported_holder.get("state")
        return target_initial if value is None else value  # type: ignore[return-value]

    def post_step() -> None:
        restored = values_after()
        target_bridge.export_checkpoint(
            root / "restored_checkpoint", restored, completed_steps=1
        )
        target_spec = target.control_specs[0].tensor
        controls = {
            "activity": torch.full(
                target_spec.shape,
                0.01,
                dtype=torch.float64,
                device=target_spec.device,
            )
        }
        imported_holder["state"] = target.step(restored, controls, 1)

    source_identity = source.identity().to_metadata()["execution"][
        "functional_runtime"
    ]
    target_identity = target.identity().to_metadata()["execution"][
        "functional_runtime"
    ]
    result = _checkpoint_common(
        cell,
        root,
        source_values=source_state,
        target_values_before=target_initial,
        target_values_after=values_after,
        checkpoint=checkpoint,
        target_probe=root / "target_probe",
        restored_checkpoint=root / "restored_checkpoint",
        tampered_checkpoint=root / "tampered_checkpoint",
        restore=restore,
        post_restore_step=post_step,
        completed_steps=1,
        fallback_used=bool(source_identity.get("fallback_used"))
        or bool(target_identity.get("fallback_used")),
    )
    return result


_EXECUTORS: dict[str, Callable[[MatrixCell, Path], dict[str, object]]] = {
    "plane_production": _run_plane,
    "periodic_production": _run_periodic,
    "channel_production": _run_channel,
    "periodic_functional": _run_functional,
    "channel_functional": _run_functional,
}


def matrix_plan() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "count": len(MATRIX),
        "cells": [cell.to_metadata() for cell in MATRIX],
        "claim_boundary": {
            "exact_serialized_restore": True,
            "finite_post_restore_step": True,
            "post_restore_cross_device_byte_identity": False,
            "performance_or_memory": False,
            "long_run": False,
        },
    }


def _validate_installed_identity(
    expected_purelib: Path | None,
    forbidden_roots: tuple[Path, ...],
) -> dict[str, object]:
    import pssolver

    imported = Path(pssolver.__file__).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if expected_purelib is not None:
        _require(purelib == expected_purelib, "active purelib differs")
        _require(imported.is_relative_to(expected_purelib), "not installed wheel")
    for root in forbidden_roots:
        _require(not imported.is_relative_to(root), "source shadow import detected")
    resolved_sys_path = [
        Path(entry or os.getcwd()).expanduser().resolve()
        for entry in sys.path
        if Path(entry or os.getcwd()).exists()
    ]
    shadow_entries = [
        str(entry)
        for entry in resolved_sys_path
        if any(
            entry == root or entry.is_relative_to(root)
            for root in forbidden_roots
        )
    ]
    _require(not shadow_entries, f"source shadow paths present: {shadow_entries}")
    return {
        "python": str(Path(sys.executable).resolve()),
        "purelib": str(purelib),
        "pssolver_file": str(imported),
        "pssolver_version": importlib.metadata.version("pssolver"),
        "source_shadow_import": False,
        "forbidden_sys_path_entries": shadow_entries,
    }


def execute(
    output_root: Path,
    *,
    expected_purelib: Path | None = None,
    forbidden_roots: tuple[Path, ...] = (),
) -> dict[str, object]:
    import torch

    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    _atomic_json(output_root / "plan.json", matrix_plan())
    identity = _validate_installed_identity(expected_purelib, forbidden_roots)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one CUDA device is required")
    _require(str(torch.empty((), device="cuda").device) == "cuda:0", "device mismatch")
    _require("H100" in torch.cuda.get_device_name(0), "allocated GPU is not H100")
    _require(not torch.backends.cuda.matmul.allow_tf32, "matmul TF32 is enabled")
    _require(not torch.backends.cudnn.allow_tf32, "cuDNN TF32 is enabled")
    environment = {
        **identity,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "allocated_device": str(torch.empty((), device="cuda").device),
        "device_name": torch.cuda.get_device_name(0),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    _atomic_json(output_root / "environment.json", environment)
    reports: list[dict[str, object]] = []
    for cell in MATRIX:
        cell_root = output_root / "cells" / cell.id
        cell_root.mkdir(parents=True)
        report_path = output_root / "reports" / f"{cell.id}.json"
        try:
            report = _EXECUTORS[cell.family](cell, cell_root)
            _atomic_json(report_path, report)
            reports.append(report)
            if report.get("passed") is not True:
                raise RuntimeError(f"{cell.id} checkpoint portability gate failed")
        except BaseException as exc:
            failure_path = output_root / "failure.json"
            if not failure_path.exists():
                _atomic_json(
                    failure_path,
                    {
                        "schema": "pssolver.rc4_2_6.failure.v1",
                        "failed_cell": cell.to_metadata(),
                        "completed_cells": [item["cell"]["id"] for item in reports],
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            raise
    completion = {
        "schema": COMPLETE_SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "cell_count": len(reports),
        "ordered_cell_ids": [report["cell"]["id"] for report in reports],
    }
    _atomic_json(output_root / "MATRIX_COMPLETE.json", completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--expected-purelib")
    parser.add_argument("--forbidden-source-root", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = Path(arguments.output)
    if arguments.plan_only:
        _atomic_json(output, matrix_plan())
        return 0
    expected = (
        None
        if arguments.expected_purelib is None
        else Path(arguments.expected_purelib).expanduser().resolve()
    )
    forbidden = tuple(
        Path(value).expanduser().resolve()
        for value in arguments.forbidden_source_root
    )
    execute(output, expected_purelib=expected, forbidden_roots=forbidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
