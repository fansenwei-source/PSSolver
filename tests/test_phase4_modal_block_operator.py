"""Phase 4.4 fixed two-component modal block qualification gates."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.core as core
import pssolver.operators as operators
from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec
from pssolver.operators.modal_block import (
    BoundTwoComponentModalOperator,
    TwoComponentModalSolveWorkspace,
    bind_two_component_modal_operator,
)
from pssolver.operators.modal_block_reference import (
    TwoComponentPeriodicReactionDiffusionReference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECLARATION_PATH = PROJECT_ROOT / "pssolver" / "core" / "modal_blocks.py"
P44_RECORD_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p44_modal_block_operator.json"
)
RUNTIME_PATHS = tuple((PROJECT_ROOT / "pssolver" / "runtime").glob("*.py"))


def _spec(**updates) -> TwoComponentModalOperatorSpec:
    values = {
        "component_order": ("u", "v"),
        "diffusion": (0.12, 0.2),
        "coupling": ((-0.3, 0.4), (-0.2, -0.1)),
    }
    values.update(updates)
    return TwoComponentModalOperatorSpec(**values)


def _model(**updates) -> TwoComponentPeriodicReactionDiffusionReference:
    values = {
        "operator_spec": _spec(),
        "point_count": 32,
        "length": 2.0 * math.pi,
        "initial_mode": 2,
        "initial_amplitudes": (0.8, -0.35),
    }
    values.update(updates)
    return TwoComponentPeriodicReactionDiffusionReference(**values)


def _bound(
    *,
    implementation: str = "closed_form_2x2",
    spec: TwoComponentModalOperatorSpec | None = None,
    eigenvalues: torch.Tensor | None = None,
) -> BoundTwoComponentModalOperator:
    declaration = spec or _spec()
    laplacian = (
        torch.tensor((0.0, -1.0, -4.0, -9.0), dtype=torch.float64)
        if eigenvalues is None
        else eigenvalues
    )
    return bind_two_component_modal_operator(
        declaration,
        laplacian,
        component_order=declaration.component_order,
        geometry_identity="periodic_1d",
        basis_signature=("periodic",),
        spectral_dtype=torch.complex128,
        implementation=implementation,
    )


def _random_rhs(mode_count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2404)
    real = torch.randn(mode_count, 2, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        mode_count,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.complex(real, imaginary)


def test_modal_block_declaration_is_tensor_and_pssolver_free():
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


def test_modal_block_declaration_freezes_component_order_and_coefficients():
    spec = _spec()
    assert spec.block_size == 2
    coefficient = spec.coefficient_matrix(-4.0)
    assert coefficient[0] == pytest.approx((-0.78, 0.4))
    assert coefficient[1] == pytest.approx((-0.2, -0.9))
    assert spec.to_metadata() == {
        "schema_version": 1,
        "identity": "two_component_modal_operator",
        "block_size": 2,
        "component_order": ["u", "v"],
        "diffusion": [0.12, 0.2],
        "coupling": [[-0.3, 0.4], [-0.2, -0.1]],
        "geometry_identity": "periodic_1d",
        "basis_signature": ["periodic"],
        "physical_coefficient_policy": "constant",
        "mode_coupling": False,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"component_order": ("u", "u")}, "component_order"),
        ({"component_order": ("u", "v", "w")}, "component_order"),
        ({"diffusion": (0.1,)}, "diffusion"),
        ({"diffusion": (0.1, -0.2)}, "non-negative"),
        ({"coupling": ((1.0, 0.0),)}, "shape"),
        ({"coupling": ((1.0, math.nan), (0.0, 1.0))}, "finite"),
        ({"geometry_identity": "plane"}, "periodic_1d"),
        ({"basis_signature": ("neumann",)}, "periodic"),
    ],
)
def test_modal_block_declaration_rejects_out_of_scope_inputs(updates, message):
    with pytest.raises(ValueError, match=message):
        _spec(**updates)


def test_binding_constructs_expected_mode_local_coefficients():
    operator = _bound()
    assert operator.coefficients.shape == (4, 2, 2)
    expected = torch.tensor(
        [
            _spec().coefficient_matrix(value)
            for value in (0.0, -1.0, -4.0, -9.0)
        ],
        dtype=torch.complex128,
    )
    assert torch.equal(operator.coefficients, expected)
    assert operator.mode_shape == (4,)
    assert operator.spec.component_order == ("u", "v")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("component_order", ("v", "u"), "component order"),
        ("geometry_identity", "plane", "geometry capability"),
        ("basis_signature", ("neumann",), "basis capability"),
    ],
)
def test_binding_rejects_capability_or_component_mismatch(
    keyword,
    value,
    message,
):
    arguments = {
        "component_order": ("u", "v"),
        "geometry_identity": "periodic_1d",
        "basis_signature": ("periodic",),
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        bind_two_component_modal_operator(
            _spec(),
            torch.tensor((0.0, -1.0), dtype=torch.float64),
            spectral_dtype=torch.complex128,
            implementation="closed_form_2x2",
            **arguments,
        )


def test_binding_rejects_dtype_nonfinite_and_unknown_implementation():
    common = {
        "component_order": ("u", "v"),
        "geometry_identity": "periodic_1d",
        "basis_signature": ("periodic",),
        "spectral_dtype": torch.complex128,
        "implementation": "closed_form_2x2",
    }
    with pytest.raises(ValueError, match="dtype is incompatible"):
        bind_two_component_modal_operator(
            _spec(),
            torch.tensor((0.0, -1.0), dtype=torch.float32),
            **common,
        )
    with pytest.raises(ValueError, match="must be finite"):
        bind_two_component_modal_operator(
            _spec(),
            torch.tensor((0.0, math.nan), dtype=torch.float64),
            **common,
        )
    with pytest.raises(ValueError, match="implementation"):
        bind_two_component_modal_operator(
            _spec(),
            torch.tensor((0.0, -1.0), dtype=torch.float64),
            **(common | {"implementation": "fallback"}),
        )


def test_closed_form_matches_cpu_linalg_reference_and_residual():
    reference = _bound(implementation="torch_linalg_reference")
    optimized = _bound(implementation="closed_form_2x2")
    rhs = _random_rhs(4)
    expected = reference.solve(rhs, alpha=3.75)
    actual = optimized.solve(rhs, alpha=3.75)
    assert torch.allclose(actual, expected, rtol=2e-15, atol=2e-15)
    identity = torch.eye(2, dtype=torch.complex128)
    residual = torch.matmul(
        3.75 * identity - optimized.coefficients,
        actual.unsqueeze(-1),
    ).squeeze(-1) - rhs
    assert float(torch.max(torch.abs(residual))) < 2e-15


def test_closed_form_workspace_is_bounded_and_reused_by_identity():
    operator = _bound()
    workspace = operator.allocate_workspace()
    rhs = _random_rhs(4)
    output_identity = workspace.output
    first = operator.solve_into(rhs, alpha=4.0, workspace=workspace).clone()
    second = operator.solve_into(2.0 * rhs, alpha=4.0, workspace=workspace)
    assert second is output_identity
    assert torch.allclose(second, 2.0 * first, rtol=2e-15, atol=2e-15)
    assert workspace.allocated_tensor_count == 5
    assert workspace.to_metadata()["bounded"] is True


def test_solve_rejects_rhs_shape_dtype_device_and_nonfinite():
    operator = _bound()
    workspace = operator.allocate_workspace()
    with pytest.raises(ValueError, match="shape or component packing"):
        operator.solve_into(
            torch.zeros(4, 3, dtype=torch.complex128),
            alpha=2.0,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="dtype"):
        operator.solve_into(
            torch.zeros(4, 2, dtype=torch.complex64),
            alpha=2.0,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="device"):
        operator.solve_into(
            torch.empty(4, 2, dtype=torch.complex128, device="meta"),
            alpha=2.0,
            workspace=workspace,
        )
    nonfinite = torch.zeros(4, 2, dtype=torch.complex128)
    nonfinite[0, 0] = complex(math.nan, 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        operator.solve_into(nonfinite, alpha=2.0, workspace=workspace)


def test_solve_rejects_singular_or_numerically_rejected_mode():
    singular = _spec(
        diffusion=(0.0, 0.0),
        coupling=((1.0, 0.0), (0.0, 2.0)),
    )
    operator = _bound(spec=singular)
    with pytest.raises(ValueError, match="singular or numerically rejected"):
        operator.solve(torch.ones(4, 2, dtype=torch.complex128), alpha=1.0)


def test_solve_rejects_rhs_alias_with_workspace_storage():
    operator = _bound()
    workspace = operator.allocate_workspace()
    with pytest.raises(ValueError, match="must not share storage"):
        operator.solve_into(
            workspace.output,
            alpha=2.0,
            workspace=workspace,
        )


def test_reference_model_has_coupled_analytic_matrix_exponential_solution():
    model = _model()
    initial = model.initial_physical(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    exact_zero = model.exact_physical(
        0.0,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    exact_later = model.exact_physical(
        0.2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert torch.equal(initial, exact_zero)
    assert torch.isfinite(exact_later).all()
    assert not torch.equal(exact_later[:, 0], initial[:, 0])
    assert not torch.equal(exact_later[:, 1], initial[:, 1])
    matrix = model.selected_mode_linear_matrix(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert matrix[0, 1] != 0.0
    assert matrix[1, 0] != 0.0


@pytest.mark.parametrize(
    "implementation",
    ["torch_linalg_reference", "closed_form_2x2"],
)
def test_reference_model_one_step_matches_direct_matrix_solve(implementation):
    model = _model()
    operator = model.bind_operator(
        dtype=torch.float64,
        device="cpu",
        implementation=implementation,
    )
    workspace = operator.allocate_workspace()
    initial = model.initial_native_spectrum(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    actual = model.implicit_euler_step(
        initial,
        dt=0.01,
        operator=operator,
        workspace=workspace,
    ).clone()
    identity = torch.eye(2, dtype=torch.complex128)
    expected = torch.linalg.solve(
        100.0 * identity - operator.coefficients,
        (initial / 0.01).unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(actual, expected, rtol=2e-15, atol=2e-15)
    assert torch.isfinite(actual).all()


def test_bound_metadata_records_optimized_gpu_capability_without_qualification():
    reference = _bound(implementation="torch_linalg_reference")
    optimized = _bound(implementation="closed_form_2x2")
    assert reference.to_metadata()["gpu_capable"] is False
    metadata = optimized.to_metadata()
    assert metadata["gpu_capable"] is True
    assert metadata["bounded_workspace"] is True
    assert metadata["coefficient_trailing_shape"] == [2, 2]
    assert metadata["component_packing"] == "trailing_component_axis_size_2"


def test_p44_types_are_direct_import_only_and_disconnected():
    names = {
        "TwoComponentModalOperatorSpec",
        "BoundTwoComponentModalOperator",
        "TwoComponentModalSolveWorkspace",
        "TwoComponentPeriodicReactionDiffusionReference",
        "bind_two_component_modal_operator",
    }
    assert names.isdisjoint(pssolver.__all__)
    assert names.isdisjoint(core.__all__)
    assert names.isdisjoint(getattr(operators, "__all__", ()))
    for path in RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "pssolver.core.modal_blocks" not in source
        assert "pssolver.operators.modal_block" not in source


def test_phase4_p44_machine_record_matches_local_contract():
    record = json.loads(P44_RECORD_PATH.read_text(encoding="utf-8"))
    assert record["status"] == (
        "P4_4_CONTRACT_EQUIVALENT_ADJUDICATION_READY"
    )
    assert record["baseline_commit"] == (
        "e262867d19f8a9ed9f30af801021cf00910542d5"
    )
    assert record["operator"] == {
        "block_size": 2,
        "coefficient_trailing_shape": [2, 2],
        "rhs_component_packing": "trailing_component_axis_size_2",
        "cpu_reference": "torch_linalg_reference",
        "optimized": "closed_form_2x2",
        "bounded_workspace_tensor_count": 5,
        "gpu_capable": True,
        "h100_qualified": False,
    }
    assert record["qualification_model"]["analytic_solution"] == (
        "matrix_exponential_single_cosine_mode"
    )
    assert record["negative_gate_categories_passed"] == 9
    assert record["connection"] == {
        "runtime_state_connected": False,
        "step_program_connected": False,
        "plane_connected": False,
        "production_default_changed": False,
        "package_root_exported": False,
        "core_package_exported": False,
        "operators_package_exported": False,
    }
    assert record["validation"] == {
        "focused_tests_passed": 131,
        "complete_tests_passed": 1802,
        "subtests_passed": 8,
        "failures": 0,
        "recovery_profiler_tests_passed": 17,
        "adjudicator_tests_passed": 13,
        "archive_sources_verified": 62,
        "git_diff_check_passed": True,
    }
    assert record["h100_attempt"] == {
        "job_id": 10837110,
        "classification": "FAIL_P4_4_MODAL_BLOCK_H100",
        "correctness_passed": True,
        "memory_passed": True,
        "small_case_performance_passed": False,
        "large_case_performance_passed": True,
        "recovery_record": (
            "notes/architecture_v0_2/phase_4_p44_h100_recovery.json"
        ),
    }
    assert record["crossover_attempt"] == {
        "job_id": 10837117,
        "classification": "FAIL_P4_4_CROSSOVER_NOT_REPRODUCED",
        "all_complete_operator_contract_gates_passed": True,
        "all_paired_trials_favored_candidate": True,
        "contract_equivalent_adjudication_pending": True,
        "adjudication_record": (
            "notes/architecture_v0_2/"
            "phase_4_p44_contract_equivalent_adjudication.json"
        ),
    }
    assert record["eligibility"] == {
        "p4_4_local_complete": True,
        "crossover_h100_scan_authorized": False,
        "analysis_only_adjudication_authorized": True,
        "p4_5_authorized": False,
        "h100_qualification_required_before_p4_5": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
