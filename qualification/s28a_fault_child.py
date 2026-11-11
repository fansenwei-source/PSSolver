"""Replay the frozen S-18 fault injection through the S-28a pressure roles."""

import hashlib
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root))
    from pssolver.engine.constraints import _FunctionalPressureRun
    from pssolver.functional.errors import FunctionalConvergenceError
    from pssolver.functional.pressure_metadata import (
        CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
        channel_pressure_solve_diagnostics,
    )
    from pssolver.linear_solvers.stokes.channel_no_slip import PCGRole
    from qualification.s28a_pcg_equivalence import StandaloneTransposeDiagnostics

    frozen = root / "benchmarks/probe_v0_4_s18_r5_postcondition.py"
    payload = frozen.read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
        "339dccb4995f66029af78a19219a4f7ee5ab617b6ab7aea037bf284d6818431a"
    ):
        raise ValueError("S-18 fault probe bytes changed")

    def functional_run(solver):
        return _FunctionalPressureRun(
            solver,
            make_diagnostics=channel_pressure_solve_diagnostics,
            error=FunctionalConvergenceError,
            schema_version=CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
        )

    source = payload.decode("utf-8")
    replacements = (
        (
            "    from pssolver.functional import ChannelPressureTransposeOperator, ChannelImplicitPressureAdjoint\n",
            "",
        ),
        ("    import pssolver.functional." + "channel_pressure as cp\n", ""),
        ("    import pssolver.functional." + "channel_pressure_adjoint as cpa\n", ""),
        ('mod = {"production": nsp, "transpose": cp, "implicit": cpa}[path]', "mod = nsp"),
        ("ChannelPressureTransposeOperator(solver)", "StandaloneTransposeDiagnostics(solver)"),
        ("rhs = tr._project_gauge(", "rhs = solver._project_pressure_gauge("),
        (
            'owner, attr = tr, "apply_pressure_operator_transpose"',
            'owner, attr = solver, "apply_pressure_operator_transpose"',
        ),
        ("ChannelImplicitPressureAdjoint(solver)", "functional_run(solver)"),
        (
            'owner, attr = implicit._operators, "apply_pressure_operator"',
            'owner, attr = solver, "_pressure_operator"',
        ),
        (
            "res = solver._solve_pressure(rhs.clone())\n"
            "                info = dict(termination=solver.last_pressure_termination_reason, "
            "acceptable=solver.last_pressure_acceptable, iterations=solver.last_pressure_iterations)",
            "res, _, report = solver.solve(rhs.clone(), dict(pressure_guess=None, "
            "pressure_guess_action=None, pressure_guess_action_exponent=None), role=PCGRole.PRODUCTION)\n"
            "                info = dict(termination=report['termination_reason'], "
            "acceptable=report['acceptable'], iterations=report['achieved_iteration_count'])",
        ),
        ("tr.solve_pressure_transpose(rhs.clone())", "tr.solve(rhs.clone())"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"frozen probe has no unique anchor: {old!r}")
        source = source.replace(old, new, 1)
    namespace = {
        "__name__": "__main__",
        "functional_run": functional_run,
        "PCGRole": PCGRole,
        "StandaloneTransposeDiagnostics": StandaloneTransposeDiagnostics,
    }
    exec(compile(source, str(frozen), "exec"), namespace)


if __name__ == "__main__":
    main()
