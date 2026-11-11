"""Replay the M1 matrix through the merged Channel solver and Engine VJP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks import verify_v0_4_s18_r5_pcg_equivalence as reference
from pssolver.engine.constraints import _FunctionalPressureRun
from pssolver.functional.errors import FunctionalConvergenceError
from pssolver.functional.pressure_metadata import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    channel_pressure_solve_diagnostics,
)
from pssolver.linear_solvers.stokes.channel_no_slip import ChannelNoSlipModalStokesSolver, PCGRole
from tests.oracles import channel_pressure_unrolled as oracle


class StandaloneTransposeDiagnostics:
    """Caller-owned diagnostics for the standalone qualification role."""

    def __init__(self, solver):
        self.solver = solver
        self._transpose_solve_sequence_id = 0
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
        )

    def _record_transpose_diagnostics(self, report):
        self.last_transpose_diagnostics = report

    def _fail_transpose_solve(
        self,
        *,
        sequence,
        rhs_norm,
        initial_residual,
        achieved_residual,
        relative_residual,
        working_exponent,
        iterations,
        termination_reason,
    ):
        report = channel_pressure_solve_diagnostics(
            self.solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="standalone_transpose_qualification",
            solve_sequence_id=sequence,
            rhs_norm=rhs_norm,
            initial_absolute_residual=initial_residual,
            achieved_absolute_residual=achieved_residual,
            achieved_normalized_residual=relative_residual,
            working_scale_binary_exponent=working_exponent,
            achieved_iteration_count=iterations,
            termination_reason=termination_reason,
        )
        self._record_transpose_diagnostics(report)
        raise FunctionalConvergenceError(
            "standalone Channel pressure transpose solve is unacceptable",
            operation="channel_pressure_transpose_solve",
            details=report.to_metadata(),
        )

    @torch.no_grad()
    def solve(self, rhs):
        return self.solver.solve_transpose(
            rhs,
            {
                "context": self,
                "action": self.solver.apply_pressure_operator_transpose,
                "diagnostics": channel_pressure_solve_diagnostics,
                "error": FunctionalConvergenceError,
                "standalone": True,
            },
            role=PCGRole.FUNCTIONAL_TRANSPOSE,
        )


def _rhs(solver, case, *, seed_offset=0):
    shape = tuple(case["shape"])
    dtype = getattr(torch, case["dtype"])
    generator = torch.Generator().manual_seed(case["seed"] + seed_offset)
    real = torch.randn((1, *shape), generator=generator, dtype=dtype)
    imag = torch.randn((1, *shape), generator=generator, dtype=dtype)
    return solver._project_pressure_gauge(torch.complex(real, imag))


def run_path(case, path):
    if path == "oracle" and case["fixed_iterations"] is None:
        return {"skipped": "oracle_requires_fixed_work"}
    _, solver = reference._build_solver(case)
    rhs = _rhs(solver, case)
    cotangent = _rhs(solver, case, seed_offset=1000)
    state = dict(
        pressure_guess=None, pressure_guess_action=None, pressure_guess_action_exponent=None
    )
    counts = {"primal": 0, "transpose": 0}
    cls = ChannelNoSlipModalStokesSolver
    primal = cls._pressure_operator
    transpose = cls.apply_pressure_operator_transpose

    def count_primal(instance, value):
        counts["primal"] += 1
        return primal(instance, value)

    def count_transpose(instance, value):
        counts["transpose"] += 1
        return transpose(instance, value)

    gradient = None
    trace = reference._DispatchTrace()
    with (
        patch.object(cls, "_pressure_operator", count_primal),
        patch.object(cls, "apply_pressure_operator_transpose", count_transpose),
        trace.mode(),
    ):
        if path == "production":
            result, state, report = solver.solve(rhs.clone(), state, role=PCGRole.PRODUCTION)
            diagnostics = {"primal": report}
        elif path == "transpose":
            context = StandaloneTransposeDiagnostics(solver)
            result = context.solve(rhs.clone())
            diagnostics = {"transpose": context.last_transpose_diagnostics.to_metadata()}
        elif path == "implicit":
            inputs = rhs.clone().requires_grad_(True)
            context = _FunctionalPressureRun(
                solver,
                make_diagnostics=channel_pressure_solve_diagnostics,
                error=FunctionalConvergenceError,
                schema_version=CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
            )
            result = context.solve(inputs)
            gradient = torch.autograd.grad(result, inputs, grad_outputs=cotangent)[0]
            diagnostics = {
                "primal": context.last_primal_diagnostics.to_metadata(),
                "transpose": context.last_transpose_diagnostics.to_metadata(),
            }
        else:
            inputs = rhs.clone().requires_grad_(True)
            with patch.object(
                oracle,
                "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
                max(oracle.CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES, rhs.numel()),
            ):
                result = oracle.unrolled_channel_pressure_solve_oracle(
                    inputs, solver, iterations=int(case["fixed_iterations"])
                )
            gradient = torch.autograd.grad(result, inputs, grad_outputs=cotangent)[0]
            diagnostics = None

    def tensor_record(value):
        return None if value is None else reference._tensor_record(value)

    return {
        "result": tensor_record(result),
        "gradient": tensor_record(gradient),
        "diagnostics": diagnostics,
        "operator_calls": counts,
        "torch_dispatch": trace.record(),
        "warm_pressure": tensor_record(state["pressure_guess"]),
        "warm_operator_action": tensor_record(state["pressure_guess_action"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rows = [
        {"case": case, "paths": {path: run_path(case, path) for path in reference.PATHS}}
        for case in reference.CASES
    ]
    args.output.write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
