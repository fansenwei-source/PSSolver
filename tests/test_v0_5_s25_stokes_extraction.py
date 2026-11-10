"""S-25 r1.2 keeps legacy MRO and gives Periodic a stateless solve."""

from __future__ import annotations

import ast
import json
import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pssolver.linear_solvers.stokes.periodic import PeriodicModalStokesSolver
from pssolver.models.active_nematics.stokes import BerisEdwardsFreeSlipStokes
from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import RuntimeCase
from qualification.identity.runtime_case_v2 import RuntimeCaseV2
from qualification.identity.scenarios import scenarios


ROOT = Path(__file__).resolve().parents[1]
BASE = "4c12228aba01e8483e80d28a1471ab72384d5fda"


def _old_method(path, class_name, name, namespace):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT).decode()
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    code = ast.get_source_segment(source, node)
    namespace = dict(namespace)
    exec(compile(code, f"{BASE}:{path}:{name}", "exec"), namespace)
    return namespace[name]


def _diagnostics(flow):
    return {
        key: tensor_record(value) if isinstance(value, torch.Tensor) else repr(value)
        for key, value in vars(flow).items()
        if key.startswith("last_")
    }


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "family,path",
    [
        ("periodic", "default"),
        ("plane", "legacy_production"),
        ("plane", "compiled_v2"),
    ],
)
def test_legacy_forward_preserves_result_diagnostics_and_mro(tmp_path, dtype, family, path):
    scene = next(row for row in scenarios() if row["id"] == f"{family}_{path}_{dtype}_default")
    if family == "periodic":
        case = RuntimeCaseV2(scene, tmp_path)
        reference = json.loads((ROOT / "tests/baselines/s25/periodic_comparison.json").read_text())[
            "core"
        ][dtype]
        flow = case.engine.constraint.term
        actual = tensor_record(
            flow.forward(case.engine.field_view(case.state), {"alpha": case.engine.default_controls["alpha"]})
        )
        assert actual == reference["static"]
        assert _diagnostics(flow) == reference["diagnostics"]
        return
    case = RuntimeCase(scene, tmp_path)
    flow = case.solver.model.static_model
    expected_forward = _old_method(
        "pssolver/models/active_nematics/stokes.py",
        "BerisEdwardsFreeSlipStokes",
        "forward",
        BerisEdwardsFreeSlipStokes.forward.__globals__,
    )
    expected_solve = _old_method(
        "pssolver/linear_solvers/stokes/periodic.py",
        "PeriodicModalStokesSolver",
        "solve_force_hats",
        PeriodicModalStokesSolver.solve_force_hats.__globals__,
    )
    with (
        patch.object(BerisEdwardsFreeSlipStokes, "forward", expected_forward),
        patch.object(PeriodicModalStokesSolver, "solve_force_hats", expected_solve),
    ):
        expected = tensor_record(case.solver.model.compute_static())
        diagnostics = _diagnostics(flow)
    actual = tensor_record(case.solver.model.compute_static())
    assert actual == expected
    assert _diagnostics(flow) == diagnostics


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("pressure_diagnostics", [False, True])
def test_periodic_stateless_solve_does_not_assign_or_extract_scalars(
    tmp_path, dtype, pressure_diagnostics
):
    scene = next(row for row in scenarios() if row["id"] == f"periodic_default_{dtype}_default")
    case = RuntimeCaseV2(scene, tmp_path)
    flow = case.engine.constraint.term
    before = dict(vars(flow))
    records = _diagnostics(flow)
    force = [case.engine.field_view(case.state)[f"{name}.hat"] for name in ("Qxx", "Qxy", "Qxz")]
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar extraction")):
        values, diagnostics = flow.solve_force_hats_with_diagnostics(
            *force, pressure_diagnostics=pressure_diagnostics
        )
    assert set(vars(flow)) == set(before)
    assert all(vars(flow)[key] is value for key, value in before.items())
    assert _diagnostics(flow) == records
    assert diagnostics["pressure_iterations"] == 1
    assert torch.equal(diagnostics["pressure_hat"], values[-1])
    if pressure_diagnostics:
        assert diagnostics["residual_norm"].ndim == diagnostics["force_norm"].ndim == 0
    else:
        assert diagnostics["residual_norm"] is diagnostics["force_norm"] is None


def test_legacy_force_recording_precedes_solver_failure_and_preserves_overrides(tmp_path):
    scene = next(
        row for row in scenarios() if row["id"] == "plane_legacy_production_float64_default"
    )
    case = RuntimeCase(scene, tmp_path)
    flow = case.solver.model.static_model
    original = flow.compute_nematic_force
    calls = []

    def force(*args):
        calls.append("force")
        return original(*args)

    def solve(*args):
        calls.append("solve")
        assert isinstance(flow.last_total_tangential_force_mean, torch.Tensor)
        raise RuntimeError("test injected solve failure")

    flow.last_total_tangential_force_mean = None
    with (
        patch.object(flow, "compute_nematic_force", force),
        patch.object(flow, "solve_force_hats", solve),
    ):
        with pytest.raises(RuntimeError, match="test injected solve failure"):
            case.solver.model.compute_static()
    assert calls == ["force", "solve"]
    assert not math.isnan(float(flow.last_total_tangential_force_mean[0]))
