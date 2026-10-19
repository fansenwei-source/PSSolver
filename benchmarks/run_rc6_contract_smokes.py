#!/usr/bin/env python3
"""Run the six RC6 public-contract smokes under the candidate H100 wheel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc6_single_h100_qualification import (
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc426_cuda_tests import Results, _enforce_tf32_policy


SCHEMA = "pssolver.rc6.contract_smokes.v1"
RELATIVE_TOLERANCE_TERMINATION_REASON = "relative_tolerance_met"
FIXED_WORK_TERMINATION_REASON = "fixed_work_completed"


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
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
            raise RuntimeError("contract-smoke JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _plane_nonfinite_cuda(root: Path) -> bool:
    import torch
    from pssolver.configuration import create_plane_beris_edwards_run_spec
    from pssolver.errors import SimulationExecutionError, SimulationStateError
    from pssolver.models.active_nematics import Q_COMPONENTS
    from pssolver.runtime.plane_compiled_v2_binding import bind_plane_compiled_v2
    from pssolver.runtime.plane_compiled_v2_step import build_plane_compiled_v2_step_program
    from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
    from pssolver.workflows.plane_compiled_v2 import PlaneCompiledV2WorkflowAdapter

    spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0, output_dir=root / "plane", device="cuda", dtype="float64",
        pointwise_execution="eager", nx=6, ny=6, nz=6, lx=8.0, ly=9.0,
        height=20.0, steps=2, save_start_step=0, save_interval=1,
        diagnostic_interval=1, diagnostics=True, spectral_refresh_steps=3,
        runtime_path="compiled_v2",
    )
    coordinate = torch.arange(6**3, dtype=torch.float64, device="cuda").reshape(6, 6, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    initial = {name: coordinate.mul((index + 1) * 1.0e-3) for index, name in enumerate(Q_COMPONENTS)}
    solver, projector = build_legacy_plane_runtime(spec, device="cuda", initial_values=initial)
    binding = bind_plane_compiled_v2(spec, solver=solver, projector=projector)
    program = build_plane_compiled_v2_step_program(binding)
    adapter = PlaneCompiledV2WorkflowAdapter(binding, program, spec)
    program.step()
    cases = {
        "normal_force": ("projected_normal_force",),
        "pressure": ("last_pressure_residual", "last_pressure_relative_residual"),
    }
    for name, expected in cases.items():
        stokes = binding.operators.stokes_kernel
        if name == "normal_force":
            stokes.last_projected_normal_force.fill_(float("nan"))
        else:
            stokes.last_projected_normal_force.zero_()
            stokes.last_pressure_residual = float("inf")
            stokes.last_pressure_relative_residual = float("nan")
        try:
            adapter.capture_diagnostic(step=1, viscosity=spec.eta, friction=0.0)
        except SimulationStateError as error:
            if not isinstance(error, (SimulationExecutionError, ValueError)):
                return False
            if error.details != {"nonfinite_fields": list(expected)}:
                return False
        else:
            return False
    return True


def _pressure_cuda() -> bool:
    import torch
    from pssolver import SpectralSolver
    from pssolver.functional import ChannelPressureTransposeOperator
    from pssolver.linear_solvers.stokes import (
        ChannelNoSlipModalStokesSolver,
        channel_pressure_minimum_relative_tolerance,
    )
    from pssolver.linear_solvers.stokes.channel_no_slip import channel_pressure_active_rms

    for dtype in (torch.float32, torch.float64):
        shape = (8, 6, 4)
        floor = channel_pressure_minimum_relative_tolerance(dtype)
        spectral = SpectralSolver(shape, L=(8.0, 6.0, 4.0), device="cuda", dtype=dtype)
        solver = ChannelNoSlipModalStokesSolver(
            spectral.transform_backend, friction=0.2, viscosity=0.73,
            pressure_relative_tolerance=floor, pressure_max_iterations=2000,
            pressure_fixed_iterations=None,
        )
        operator = ChannelPressureTransposeOperator(solver)
        complex_dtype = torch.complex64 if dtype is torch.float32 else torch.complex128
        generator = torch.Generator(device="cuda").manual_seed(9)
        rhs = operator._project_gauge(torch.complex(
            torch.randn((1, *shape), generator=generator, dtype=dtype, device="cuda"),
            torch.randn((1, *shape), generator=generator, dtype=dtype, device="cuda"),
        ).to(complex_dtype))
        pressure = solver._solve_pressure(rhs.clone())
        residual = operator._project_gauge(rhs - operator.apply_pressure_operator(pressure))
        recomputed = (
            channel_pressure_active_rms(residual, solver.pressure_inactive_mask).float64_value
            / channel_pressure_active_rms(rhs, solver.pressure_inactive_mask).float64_value
        )
        if (
            solver.last_pressure_termination_reason
            != RELATIVE_TOLERANCE_TERMINATION_REASON
        ):
            return False
        if solver.last_pressure_relative_residual > floor or recomputed > floor:
            return False
        fixed = ChannelNoSlipModalStokesSolver(
            spectral.transform_backend, friction=0.2, viscosity=0.73,
            pressure_relative_tolerance=floor / 8.0, pressure_max_iterations=4,
            pressure_fixed_iterations=4,
        )
        fixed_pressure = fixed._solve_pressure(rhs.clone())
        if fixed.last_pressure_termination_reason != FIXED_WORK_TERMINATION_REASON:
            return False
        if not bool(torch.isfinite(fixed_pressure).all().item()):
            return False
    return True


def execute(test_root: Path, work_root: Path, output: Path) -> dict[str, object]:
    import importlib.metadata
    import pytest
    import torch

    policy = _enforce_tf32_policy()
    if policy.get("enforcement", {}).get("passed") is not True:
        raise RuntimeError("TF32 policy enforcement failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("exactly one H100 is required")
    if importlib.metadata.version("pssolver") != "0.2.0rc6":
        raise RuntimeError("candidate package version differs")
    support = test_root.expanduser().resolve()
    if not support.is_dir() or support.is_symlink() or (support / "pssolver").exists():
        raise RuntimeError("ordinary non-shadowing test root required")
    root = work_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    checks = {
        "plane_compiled_nonfinite_public_error_on_cuda": _plane_nonfinite_cuda(root),
        "pressure_floor_fixed_and_tolerance_modes_on_cuda": _pressure_cuda(),
    }
    collector = Results()
    previous = Path.cwd()
    nodes = [
        "tests/test_rc6_3_compile_preflight_contract.py",
        "tests/test_rc6_rc64_channel_failure_contract.py",
        "tests/test_rc6_rc65_checkpoint_control_flow.py",
        "tests/test_phase2_plane_workflow_migration.py::test_restart_integrity_rejection_does_not_create_output",
    ]
    try:
        os.chdir(support)
        exit_code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *nodes], plugins=[collector]))
    finally:
        os.chdir(previous)
    outcomes = collector.metadata()
    if exit_code != 0 or any(outcomes[name] for name in ("failed", "skipped", "xfailed", "xpassed", "collection_errors")):
        raise RuntimeError("RC6 public contract pytest gate failed")
    passed = [str(value) for value in outcomes["passed"]]
    checks.update({
        "complete_stress_compile_preflight_before_allocation": any("test_rc6_3_compile_preflight_contract.py" in value for value in passed),
        "channel_public_failure_hierarchy_and_metadata": any("test_rc6_rc64_channel_failure_contract.py" in value for value in passed),
        "checkpoint_early_stop_not_translated": any("test_rc6_rc65_checkpoint_control_flow.py" in value for value in passed),
        "rejected_restart_does_not_materialize_output": any("test_restart_integrity_rejection_does_not_create_output" in value for value in passed),
    })
    expected = build_plan()["rc6_contract_smokes"]
    if checks != expected:
        raise RuntimeError(f"RC6 contract-smoke set failed: {checks}")
    result = {
        "schema": SCHEMA,
        "provider_commit": PROVIDER_CANDIDATE,
        "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        "passed": True,
        "checks": checks,
        "pytest_passed_count": len(passed),
        "tf32_policy": policy,
    }
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.test_root, args.work_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
