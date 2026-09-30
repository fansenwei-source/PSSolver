"""Static closure and Channel-functional planning gates for P9.6--P9.7.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
)
from pssolver.models.active_nematics.channel_stokes import (
    CHANNEL_Q_BOUNDARY_CONDITIONS,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _record(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p96_closure_authorizes_planning_but_not_channel_implementation():
    record = _record("phase_9_p96_h100_closure.json")
    assert record["classification"] == (
        "PASS_P9_6_INDEPENDENT_PERIODIC_CONSUMER_INSTALLED_WHEEL_H100"
    )
    assert record["provider_commit"] == (
        "d878abd3caecffb99f2cbf9255963a42f401af11"
    )
    assert record["gradient_contract"]["authoritative_gate"] == (
        "independent_direction_gradient_subtracted_taylor_second_order"
    )
    assert record["gradient_contract"]["all_second_order_gates_passed"] is True
    assert record["archive"]["checksum_passed"] == 24
    assert record["authorization"]["p9_6_complete"] is True
    assert record["authorization"]["p9_7_planning_eligible"] is True
    assert record["authorization"]["p9_7_implementation_authorized"] is False


def test_p970_first_combination_matches_existing_channel_basis_contract():
    record = _record("phase_9_p970_channel_functional_plan.json")
    combination = record["first_supported_combination"]
    assert combination["model"] == "complete_stress_beris_edwards"
    assert combination["geometry"] == "rectangular_channel"
    assert combination["periodic_axes"] == ["x"]
    assert combination["bounded_axes"] == ["y", "z"]
    assert combination["boundary_conditions"]["Q"] == list(
        CHANNEL_Q_BOUNDARY_CONDITIONS
    )
    assert combination["boundary_conditions"]["velocity"] == list(
        CHANNEL_VELOCITY_BOUNDARY_CONDITIONS
    )
    assert combination["boundary_conditions"]["pressure"] == list(
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS
    )
    assert combination["pressure_gauge"] == "zero_mean"
    assert combination["batch_sizes"] == [1]


def test_p970_freezes_pure_state_and_implicit_pressure_adjoint():
    record = _record("phase_9_p970_channel_functional_plan.json")
    contract = record["functional_contract"]
    assert contract["state_components"] == ["q_physical", "q_spectral"]
    assert contract["pressure_guess_in_state"] is False
    assert contract["pressure_warm_start_policy"] == "disabled_in_functional_path"
    assert contract["production_pressure_warm_start_unchanged"] is True
    assert contract["input_mutation_allowed"] is False
    assert contract["pure_step_required"] is True

    pressure = record["pressure_adjoint_contract"]
    assert pressure["production_strategy"] == "custom_implicit_adjoint"
    assert pressure["transpose_symmetry_assumed_without_test"] is False
    assert pressure["forward_iterations_retained_for_backward"] is False
    assert pressure["unrolled_fixed_iteration_role"] == (
        "small_grid_cpu_oracle_only"
    )
    for gate in (
        "transpose_dot_product_identity",
        "manufactured_transpose_pressure_solve",
        "implicit_vjp_vs_unrolled_small_grid_oracle",
        "memory_independent_of_pcg_iteration_history",
    ):
        assert gate in pressure["required_gates"]


def test_p970_preserves_provider_consumer_and_legacy_oracle_boundaries():
    record = _record("phase_9_p970_channel_functional_plan.json")
    ownership = record["ownership"]
    assert ownership["dependency_direction"] == (
        "independent_consumer_to_pssolver_public_api_only"
    )
    assert ownership["pssolver_imports_consumer"] is False
    assert ownership["consumer_private_pssolver_access"] is False
    assert ownership["legacy_pssolver_control_status"] == (
        "frozen_oracle_no_new_features"
    )
    consumer = record["second_consumer"]
    assert consumer["must_use_installed_wheel"] is True
    assert "private_pssolver_members" in consumer["must_not_use"]
    assert "pssolver.control_as_runtime_dependency" in consumer["must_not_use"]


def test_p970_inventory_hashes_bind_the_audited_sources():
    record = _record("phase_9_p970_channel_functional_plan.json")
    for entry in record["existing_assets"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert len(entry["sha256"]) == 64
        assert len(entry["git_blob"]) == 40
        if entry["change_policy"] == "frozen_oracle":
            assert _sha256(path) == entry["sha256"]
        else:
            assert entry["change_policy"] == "p9_7_subject_to_reviewed_change"


def test_p970_slices_are_ordered_and_later_work_remains_unauthorized():
    record = _record("phase_9_p970_channel_functional_plan.json")
    assert [item["id"] for item in record["slices"]] == [
        "P9.7.1",
        "P9.7.2",
        "P9.7.3",
        "P9.7.4",
        "P9.7.5",
        "P9.7.6",
    ]
    authorization = record["authorization"]
    assert authorization["p9_7_0_complete"] is True
    assert authorization["p9_7_1_planning_eligible"] is True
    assert authorization["p9_7_1_implementation_authorized"] is False
    assert authorization["p9_7_2_through_p9_7_6_authorized"] is False
    assert authorization["phase_9_complete"] is False
    assert authorization["production_default_changed"] is False


def test_verbatim_archive_tracks_p96_closure_and_p970_without_rendering_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    for name in (
        "phase_9_p96_h100_closure.md",
        "phase_9_p96_h100_closure.json",
        "phase_9_p970_channel_functional_plan.md",
        "phase_9_p970_channel_functional_plan.json",
    ):
        assert f'"{name}"' in source
    assert _record("phase_9_p96_h100_closure.json")["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
    assert _record("phase_9_p970_channel_functional_plan.json")["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
