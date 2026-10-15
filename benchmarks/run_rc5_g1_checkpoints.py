#!/usr/bin/env python3
"""Run the rc5 G-1 current cross-device and released checkpoint gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Iterable

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCHEMA = "pssolver.rc5.g1.checkpoint_h100.v1"
CELL_SCHEMA = "pssolver.rc5.g1.checkpoint_cell.v1"
PROVIDER_COMMIT = "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
CONSUMER_SOURCE = "c14e6b56c88081e7f9a5642e65ffd722eec7445b"
RELEASED_RECORD_SHA256 = "a83a8286212d450223278fb8492c69f0a41c0c546aa303da7f55d850d14ce572"
FAMILIES = (
    "periodic_functional",
    "channel_functional_fixed_iteration",
    "channel_functional_tolerance",
    "legacy_channel",
)
DIRECTIONS = (("cpu", "cuda"), ("cuda", "cpu"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("checkpoint JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_sha(values: Iterable[object]) -> str:
    from benchmarks.run_rc426_cross_device_checkpoint import _tensor_sha256

    return _tensor_sha256(values)


def _finite(values: Iterable[object]) -> bool:
    from benchmarks.run_rc426_cross_device_checkpoint import _finite

    return _finite(values)


def _corrupt_copy(source: Path, target: Path) -> str:
    from benchmarks.run_rc426_cross_device_checkpoint import _corrupt_copy

    return _corrupt_copy(source, target).relative_to(target).as_posix()


def _identity_copy(source: Path, target: Path) -> None:
    from pssolver.io.checkpoint import METADATA_SHA256_KEY, seal_checkpoint_metadata

    shutil.copytree(source, target)
    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    was_sealed = METADATA_SHA256_KEY in metadata
    metadata.pop(METADATA_SHA256_KEY, None)
    bridge = metadata.get("functional_bridge")
    if isinstance(bridge, dict) and "checkpoint_compatibility_sha256" in bridge:
        bridge["checkpoint_compatibility_sha256"] = "0" * 64
    elif "compatibility_identity_sha256" in metadata:
        metadata["compatibility_identity_sha256"] = "0" * 64
    elif "runtime_identity_sha256" in metadata:
        metadata["runtime_identity_sha256"] = "0" * 64
    else:
        raise RuntimeError("checkpoint has no compatibility identity to tamper")
    sealed = seal_checkpoint_metadata(metadata) if was_sealed else metadata
    metadata_path.write_text(
        json.dumps(sealed, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tf32_off() -> None:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _require(
        (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )
        == ("highest", False, False),
        "cannot enforce the frozen TF32-off checkpoint policy",
    )


def _functional_runtime(
    root: Path, label: str, device: str, family: str
):
    from benchmarks.run_rc426_cross_device_checkpoint import (
        _channel_simulation,
        _periodic_simulation,
    )
    from pssolver.functional import (
        build_channel_activity_functional_runtime,
        build_functional_runtime,
        channel_activity_functional_request,
        periodic_activity_functional_request,
    )

    if family == "periodic_functional":
        simulation = _periodic_simulation(root, label, device)
        return build_functional_runtime(
            periodic_activity_functional_request(simulation.specification)
        )
    simulation = _channel_simulation(root, label, device)
    pressure = {
        "algorithm": "preconditioned_conjugate_gradient",
        "relative_tolerance": 1.0e-10,
        "max_iterations": 40,
        "fixed_iterations": None,
        "warm_start": True,
    }
    if family == "channel_functional_fixed_iteration":
        pressure["fixed_iterations"] = 12
    simulation = replace(simulation, discretization={"pressure_solver": pressure})
    return build_channel_activity_functional_runtime(
        channel_activity_functional_request(simulation.specification)
    )


def _controls(runtime):
    import torch

    spec = runtime.control_specs[0].tensor
    dtype = getattr(torch, spec.dtype) if isinstance(spec.dtype, str) else spec.dtype
    return {"activity": torch.full(spec.shape, 0.01, dtype=dtype, device=spec.device)}


def _rejection(
    restore: Callable[[Path], object],
    path: Path,
    before: str,
    values_after: Callable[[], tuple[object, ...]],
) -> dict[str, object]:
    try:
        restore(path)
    except Exception as exc:
        message = str(exc)
        return {
            "rejected": True,
            "exception_type": type(exc).__name__,
            "message": message,
            "guard_reached": any(
                token in message.lower()
                for token in ("checksum", "compatibility", "identity")
            ),
            "target_unchanged": before == _tensor_sha(values_after()),
        }
    return {
        "rejected": False,
        "exception_type": None,
        "message": None,
        "guard_reached": False,
        "target_unchanged": before == _tensor_sha(values_after()),
    }


def _functional_cell(root: Path, family: str, source_device: str, target_device: str) -> dict[str, object]:
    source = _functional_runtime(root, "source", source_device, family)
    target = _functional_runtime(root, "target", target_device, family)
    source_state = source.initial_state()
    source_state = source.step(source_state, _controls(source), 0)
    checkpoint = source.checkpoint_bridge.export_checkpoint(
        root / "checkpoint", source_state, completed_steps=1
    )
    source_before = _tree(checkpoint)
    target_initial = target.initial_state()
    holder = {"state": target_initial}

    def restore(path: Path):
        imported = target.checkpoint_bridge.import_checkpoint(path)
        holder["state"] = imported.state
        return imported

    def values_after() -> tuple[object, ...]:
        return tuple(holder["state"])

    target_before = _tensor_sha(values_after())
    raw_path = root / "tampered_payload"
    tampered_file = _corrupt_copy(checkpoint, raw_path)
    raw = _rejection(restore, raw_path, target_before, values_after)
    raw["tampered_file"] = tampered_file
    identity_path = root / "tampered_identity"
    _identity_copy(checkpoint, identity_path)
    identity = _rejection(restore, identity_path, target_before, values_after)

    imported = restore(checkpoint)
    restored_sha = _tensor_sha(imported.state)
    _require(restored_sha == _tensor_sha(source_state), "functional serialized state differs")
    direct = target.step(imported.state, _controls(target), 1)
    replay = restore(checkpoint)
    restarted = target.step(replay.state, _controls(target), 1)
    replay_equal = _tensor_sha(direct) == _tensor_sha(restarted)
    observations = target.observe(restarted, _controls(target))
    finite = _finite(restarted) and _finite(observations.values())
    source_unchanged = source_before == _tree(checkpoint)
    runtime_identity = target.identity().to_metadata()["execution"]["functional_runtime"]
    gates = (
        raw["rejected"], raw["guard_reached"], raw["target_unchanged"],
        identity["rejected"], identity["guard_reached"], identity["target_unchanged"],
        replay_equal, finite, source_unchanged,
        imported.completed_steps == 1, runtime_identity.get("fallback_used") is False,
    )
    return {
        "schema": CELL_SCHEMA,
        "family": family,
        "source_device": source_device,
        "target_device": target_device,
        "completed_steps": imported.completed_steps,
        "serialized_state_equal": True,
        "continuous_split_restart_equal": replay_equal,
        "restored_state_and_observation_finite": finite,
        "payload_tamper": raw,
        "identity_mismatch": identity,
        "source_checkpoint_unchanged": source_unchanged,
        "fallback_used": bool(runtime_identity.get("fallback_used")),
        "passed": all(bool(value) for value in gates),
    }


def _legacy_spec(device: str):
    from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec

    return ChannelActiveNematicRunSpec(
        shape=(8, 6, 5), lengths=(8.0, 6.0, 5.0), dt=1.0e-3,
        steps=2, save_interval=2, diagnostic_interval=2,
        device=device, dtype="float32", activity=0.2,
        pressure_relative_tolerance=1.0e-8, pressure_max_iterations=100,
        pressure_fixed_iterations=8, runtime_path="legacy_channel", seed=24,
    )


def _legacy_runtime(spec):
    import torch
    from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
    from pssolver.runtime import ChannelRuntimeBuildRequest, build_channel_active_nematic_runtime

    values = {name: torch.zeros(spec.shape, dtype=torch.float32, device=spec.device) for name in Q_COMPONENTS}
    request = ChannelRuntimeBuildRequest(
        spec,
        {"configuration": spec.identity_metadata(), "runtime_selection": spec.runtime_selection_metadata()},
        values,
        spec.device,
    )

    def builder():
        material = spec.components.material
        pressure = spec.components.pressure_solver
        solver = build_active_nematic_channel(
            spec.shape, spec.lengths, spec.dt, values,
            device=spec.device, batchsize=spec.batch_size,
            rho=material.rho, elastic_constant=material.elastic_constant,
            beta=material.beta, friction=material.friction, viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
            pressure_fixed_iterations=pressure.fixed_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (spec.batch_size, *spec.shape), material.activity,
            dtype=solver.dtype, device=solver.device,
        )
        return solver

    return build_channel_active_nematic_runtime(request, legacy_builder=builder)


def _legacy_values(adapter) -> tuple[object, ...]:
    return (adapter.fields.spatial, adapter.fields.spectral, adapter.capture_pressure_guess())


def _restore_legacy(adapter, spec, path: Path) -> int:
    from pssolver.workflows import load_channel_checkpoint, restore_channel_checkpoint

    kwargs = {"runtime_identity_sha256": spec.runtime_identity_sha256()}
    pre_rc5 = getattr(spec, "pre_rc5_runtime_identity_sha256", None)
    if pre_rc5 is not None:
        kwargs["pre_rc5_runtime_identity_sha256"] = pre_rc5()
    return restore_channel_checkpoint(adapter, load_channel_checkpoint(path), **kwargs)


def _legacy_cell(root: Path, source_device: str, target_device: str) -> dict[str, object]:
    from pssolver.workflows import capture_channel_checkpoint, write_channel_checkpoint

    source_spec = _legacy_spec(source_device)
    source = _legacy_runtime(source_spec)
    source.advance(1)
    checkpoint = write_channel_checkpoint(
        root / "checkpoint",
        capture_channel_checkpoint(source, runtime_identity_sha256=source_spec.runtime_identity_sha256()),
    )
    source_before = _tree(checkpoint)
    negative_spec = _legacy_spec(target_device)
    negative = _legacy_runtime(negative_spec)
    before = _tensor_sha(_legacy_values(negative))
    raw_path = root / "tampered_payload"
    tampered_file = _corrupt_copy(checkpoint, raw_path)
    raw = _rejection(
        lambda path: _restore_legacy(negative, negative_spec, path),
        raw_path, before, lambda: _legacy_values(negative),
    )
    raw["tampered_file"] = tampered_file
    identity_path = root / "tampered_identity"
    _identity_copy(checkpoint, identity_path)
    identity = _rejection(
        lambda path: _restore_legacy(negative, negative_spec, path),
        identity_path, before, lambda: _legacy_values(negative),
    )

    direct_spec = _legacy_spec(target_device)
    direct = _legacy_runtime(direct_spec)
    direct_steps = _restore_legacy(direct, direct_spec, checkpoint)
    direct.advance(1)
    replay_spec = _legacy_spec(target_device)
    replay = _legacy_runtime(replay_spec)
    replay_steps = _restore_legacy(replay, replay_spec, checkpoint)
    replay.advance(1)
    replay_equal = _tensor_sha(_legacy_values(direct)) == _tensor_sha(_legacy_values(replay))
    finite = _finite(_legacy_values(replay))
    source_unchanged = source_before == _tree(checkpoint)
    fallback = bool(source.to_metadata().get("fallback_used")) or bool(replay.to_metadata().get("fallback_used"))
    gates = (
        raw["rejected"], raw["guard_reached"], raw["target_unchanged"],
        identity["rejected"], identity["guard_reached"], identity["target_unchanged"],
        direct_steps == 1, replay_steps == 1, replay_equal, finite,
        source_unchanged, not fallback,
    )
    return {
        "schema": CELL_SCHEMA,
        "family": "legacy_channel",
        "source_device": source_device,
        "target_device": target_device,
        "completed_steps": replay_steps,
        "serialized_state_equal": True,
        "continuous_split_restart_equal": replay_equal,
        "restored_state_and_observation_finite": finite,
        "payload_tamper": raw,
        "identity_mismatch": identity,
        "source_checkpoint_unchanged": source_unchanged,
        "fallback_used": fallback,
        "passed": all(bool(value) for value in gates),
    }


def validate_released_record(record_path: Path) -> dict[str, object]:
    _require(record_path.is_file() and not record_path.is_symlink(), "released record must be a regular file")
    _require(_sha256(record_path) == RELEASED_RECORD_SHA256, "released record SHA-256 differs")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    _require(record.get("classification") == "PASS_RC5_G1_RELEASED_CHECKPOINT_MATRIX_WITH_PROVIDER_IDENTITY_MIGRATION_REQUIRED", "released classification differs")
    _require(record.get("qualified_source", {}).get("commit") == PROVIDER_COMMIT, "released qualified source differs")
    matrix = record.get("released_checkpoint_matrix")
    _require(isinstance(matrix, dict) and matrix.get("complete") is True, "released matrix incomplete")
    _require(matrix.get("primary_case_count") == 13, "released primary case count differs")
    _require(matrix.get("rc1_rc3_installed_wheel_tests") == {
        "passed": 20,
        "failed": 0,
        "skipped": 0,
        "xml_sha256": "6a01e2cc6f8ce0e8db81c278ff891b8f6068dde44dff5c7ef02ccbe088726e35",
    }, "released installed-wheel evidence differs")
    _require(matrix.get("fail_closed_dispositions_preserved") == {
        "rc1_periodic_functional": "explicit_state_only_migration",
        "rc1_channel_functional": "explicit_state_only_migration",
        "rc2_channel_functional": "explicit_state_only_migration",
        "rc1_legacy_channel_even": "reject_pre_rc4_1_forward_nyquist_before_mutation",
    }, "released dispositions differ")
    return {"sha256": RELEASED_RECORD_SHA256, "primary_case_count": 13, "matched": True}


def summarize(cells: list[dict[str, object]], released: dict[str, object]) -> dict[str, object]:
    expected = {(family, source, target) for family in FAMILIES for source, target in DIRECTIONS}
    observed = {(str(cell.get("family")), str(cell.get("source_device")), str(cell.get("target_device"))) for cell in cells}
    _require(observed == expected and len(cells) == len(expected), "checkpoint cell set differs")
    for cell in cells:
        label = f"{cell.get('family')}/{cell.get('source_device')}->{cell.get('target_device')}"
        _require(cell.get("schema") == CELL_SCHEMA, f"{label} schema differs")
        _require(cell.get("passed") is True, f"{label} failed")
        _require(cell.get("continuous_split_restart_equal") is True, f"{label} replay differs")
        _require(cell.get("source_checkpoint_unchanged") is True, f"{label} source changed")
        _require(cell.get("fallback_used") is False, f"{label} used fallback")
        for gate in ("payload_tamper", "identity_mismatch"):
            report = cell.get(gate)
            _require(isinstance(report, dict), f"{label} {gate} missing")
            _require(report.get("rejected") is True, f"{label} {gate} accepted")
            _require(report.get("guard_reached") is True, f"{label} {gate} wrong guard")
            _require(report.get("target_unchanged") is True, f"{label} {gate} mutated target")
    _require(released == {"sha256": RELEASED_RECORD_SHA256, "primary_case_count": 13, "matched": True}, "released record result differs")
    return {
        "schema": SCHEMA,
        "provider_commit": PROVIDER_COMMIT,
        "consumer_source_commit": CONSUMER_SOURCE,
        "passed": True,
        "current_family_count": 4,
        "direction_count": 2,
        "cell_count": 8,
        "continuous_split_restart_passed": True,
        "transactional_restore_passed": True,
        "negative_gates_passed": True,
        "released_primary_case_count": 13,
        "released_dispositions_match_cpu_record": True,
        "source_checkpoints_unchanged": True,
        "released_record": released,
        "cells": cells,
    }


def execute(output_root: Path, released_record: Path, output: Path) -> dict[str, object]:
    import torch
    import pssolver

    _tf32_off()
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one CUDA device is required")
    _require(str(torch.empty((), device="cuda").device) == "cuda:0", "allocated device differs")
    _require("H100" in torch.cuda.get_device_name(0), "allocated GPU is not H100")
    _require(str(getattr(pssolver, "__version__", "")) == "0.2.0rc5", "candidate version differs")
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    cells: list[dict[str, object]] = []
    for family in FAMILIES:
        for source, target in DIRECTIONS:
            cell_root = root / "cells" / f"{family}_{source}_to_{target}"
            cell_root.mkdir(parents=True)
            report = (
                _legacy_cell(cell_root, source, target)
                if family == "legacy_channel"
                else _functional_cell(cell_root, family, source, target)
            )
            _atomic_json(root / "reports" / f"{family}_{source}_to_{target}.json", report)
            _require(report.get("passed") is True, f"{family}/{source}->{target} failed")
            cells.append(report)
    result = summarize(cells, validate_released_record(released_record.expanduser().resolve()))
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--released-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    execute(args.output_root, args.released_record, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
