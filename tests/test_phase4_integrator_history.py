"""Phase 4.1 tensor-free declarations and disconnected history gates."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.core as core
import pssolver.integrators as integrators
from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.history import SBDF2History


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECLARATION_PATH = PROJECT_ROOT / "pssolver" / "core" / "integrators.py"
HISTORY_PATH = PROJECT_ROOT / "pssolver" / "integrators" / "history.py"
P41_RECORD_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p41_integrator_history.json"
)
RUNTIME_PATHS = tuple((PROJECT_ROOT / "pssolver" / "runtime").glob("*.py"))


def test_integrator_declaration_is_tensor_and_pssolver_free():
    tree = ast.parse(DECLARATION_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "numpy" not in imported_roots
    assert "pssolver" not in imported_roots


def test_integrator_specs_freeze_scheme_order_and_startup():
    euler = IntegratorSpec.projected_semi_implicit_euler(dt=0.01)
    sbdf2 = IntegratorSpec.sbdf2(dt=0.01)
    assert euler.scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    assert euler.formal_order == 1
    assert euler.history_depth == 0
    assert euler.startup_scheme is None
    assert sbdf2.scheme is IntegratorScheme.SBDF2
    assert sbdf2.formal_order == 2
    assert sbdf2.history_depth == 1
    assert sbdf2.startup_scheme is (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    )
    assert sbdf2.constant_step is True
    assert sbdf2.to_metadata() == {
        "schema_version": 1,
        "scheme": "sbdf2",
        "dt": 0.01,
        "formal_order": 2,
        "history_depth": 1,
        "constant_step": True,
        "startup_scheme": "projected_semi_implicit_euler",
    }


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, -math.inf, math.nan, True])
def test_integrator_spec_rejects_invalid_dt(value):
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        IntegratorSpec.sbdf2(dt=value)


def test_integrator_spec_rejects_incompatible_startup_identity():
    with pytest.raises(ValueError, match="requires projected"):
        IntegratorSpec(
            scheme=IntegratorScheme.SBDF2,
            dt=0.01,
            startup_scheme=IntegratorScheme.SBDF2,
        )
    with pytest.raises(ValueError, match="has no startup"):
        IntegratorSpec(
            scheme=IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER,
            dt=0.01,
            startup_scheme=(
                IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            ),
        )


def test_sbdf2_history_preserves_supplied_tensor_identity():
    previous = torch.zeros((2, 3, 4), dtype=torch.complex128)
    rhs = torch.ones_like(previous)
    history = SBDF2History(previous, rhs, source_completed_steps=4, dt=0.01)
    assert history.previous_evolved_native_spectrum is previous
    assert history.previous_explicit_native_spectral_rhs is rhs
    current = torch.full_like(previous, 2.0)
    history.validate_for_current(current, completed_steps=5, dt=0.01)
    assert history.to_metadata() == {
        "schema_version": 1,
        "scheme": "sbdf2",
        "history_depth": 1,
        "source_completed_steps": 4,
        "dt": 0.01,
        "stores_tensors_by_identity": True,
        "previous_evolved_native_spectrum": {
            "shape": [2, 3, 4],
            "dtype": "torch.complex128",
            "device": "cpu",
        },
        "previous_explicit_native_spectral_rhs": {
            "shape": [2, 3, 4],
            "dtype": "torch.complex128",
            "device": "cpu",
        },
    }


def test_sbdf2_history_rejects_shared_or_incompatible_storage():
    previous = torch.zeros((2, 3), dtype=torch.complex128)
    with pytest.raises(ValueError, match="must not share storage"):
        SBDF2History(previous, previous, source_completed_steps=0, dt=0.01)
    with pytest.raises(ValueError, match="must not share storage"):
        SBDF2History(
            previous,
            previous.view_as(previous),
            source_completed_steps=0,
            dt=0.01,
        )
    with pytest.raises(ValueError, match="storage must match"):
        SBDF2History(
            previous,
            torch.zeros((2, 4), dtype=torch.complex128),
            source_completed_steps=0,
            dt=0.01,
        )
    with pytest.raises(ValueError, match="storage must match"):
        SBDF2History(
            previous,
            torch.zeros((2, 3), dtype=torch.complex64),
            source_completed_steps=0,
            dt=0.01,
        )


def test_sbdf2_history_rejects_wrong_clock_dt_and_current_layout():
    previous = torch.zeros((2, 3), dtype=torch.complex128)
    rhs = torch.ones_like(previous)
    history = SBDF2History(previous, rhs, source_completed_steps=7, dt=0.01)
    with pytest.raises(ValueError, match="immediately precede"):
        history.validate_for_current(
            torch.empty_like(previous),
            completed_steps=7,
            dt=0.01,
        )
    with pytest.raises(ValueError, match="changing dt"):
        history.validate_for_current(
            torch.empty_like(previous),
            completed_steps=8,
            dt=0.02,
        )
    with pytest.raises(ValueError, match="incompatible with history"):
        history.validate_for_current(
            torch.empty((2, 4), dtype=torch.complex128),
            completed_steps=8,
            dt=0.01,
        )


def test_phase4_p41_types_are_direct_import_only_and_disconnected():
    names = {"IntegratorScheme", "IntegratorSpec", "SBDF2History"}
    assert names.isdisjoint(pssolver.__all__)
    assert names.isdisjoint(core.__all__)
    assert names.isdisjoint(integrators.__all__)
    for path in RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "pssolver.core.integrators" not in source
        assert "pssolver.integrators.history" not in source


def test_phase4_p41_machine_record_matches_disconnected_contract():
    record = json.loads(P41_RECORD_PATH.read_text(encoding="utf-8"))
    assert record["status"] == "P4_1_LOCAL_COMPLETE_DISCONNECTED"
    assert record["baseline_commit"] == (
        "6ba9ccda84922d629cdb6d75fd39d14ecffbd0fe"
    )
    assert record["connection"] == {
        "runtime_state_connected": False,
        "step_program_connected": False,
        "plane_connected": False,
        "checkpoint_v1_changed": False,
        "package_root_exported": False,
        "core_package_exported": False,
        "integrators_package_exported": False,
    }
    assert record["validation"] == {
        "focused_tests_passed": 29,
        "complete_tests_passed": 1685,
        "subtests_passed": 8,
        "failures": 0,
        "archive_sources_verified": 52,
        "git_diff_check_passed": True,
    }
    assert record["eligibility"] == {
        "p4_1_complete": True,
        "p4_2_authorized": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
