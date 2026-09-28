"""Static contracts for the planning-only Phase 9 P9.0 freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver.control as legacy_control


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p90_control_contract_freeze.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def test_p90_binds_phase8_and_authorizes_only_the_contract_freeze() -> None:
    record = _record()
    baseline = record["baseline"]
    authorization = record["authorization"]

    assert record["schema_version"] == 1
    assert record["phase"] == "P9.0"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_0_CONTROL_CONTRACT_AND_ORACLE_FREEZE"
    )
    assert baseline["commit"] == (
        "4688021624b017fd55e476466310e4908cd19b99"
    )
    assert _sha256(ROOT / baseline["phase_8_record"]) == baseline[
        "phase_8_record_sha256"
    ]
    assert baseline["phase_8_complete"] is True
    assert baseline["eligible_for_phase_9_planning"] is True
    assert authorization == {
        "p9_0_complete": True,
        "p9_1_implementation_authorized": False,
        "later_phase_9_slices_authorized": False,
        "h100_authorized": False,
        "runtime_or_timestep_changed": False,
        "equation_or_operator_changed": False,
        "checkpoint_or_output_schema_changed": False,
        "public_package_root_changed": False,
        "production_default_changed": False,
        "verbatim_pdf_regenerated": False,
    }
    assert record["local_verification"] == {
        "focused": "74 passed",
        "full": "2436 passed, 8 subtests passed",
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "git_diff_check": "pass",
        "h100_submissions": 0,
        "verbatim_pdf_regenerated": False,
    }


def test_p90_freezes_r1_through_r12_and_their_first_owners() -> None:
    requirements = _record()["requirements"]
    by_id = {item["id"]: item for item in requirements}

    assert list(by_id) == [f"R{index}" for index in range(1, 13)]
    assert {
        identifier
        for identifier, item in by_id.items()
        if item["first_periodic_slice_blocking"]
    } == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R10", "R12"}
    assert by_id["R1"]["first_owner"] == "P9.1"
    assert by_id["R3"]["first_owner"] == "P9.2"
    assert by_id["R5"]["first_owner"] == "P9.3"
    assert by_id["R12"]["first_owner"] == "P9.4"
    assert by_id["R11"]["first_owner"] == "P9.7"
    assert all(item["acceptance"] for item in requirements)


def test_p90_freezes_ownership_and_one_way_dependency() -> None:
    record = _record()
    dependency = record["dependency_direction"]
    ownership = record["ownership"]

    assert dependency == {
        "allowed": (
            "independent_control_consumer_to_pssolver_public_provisional_api"
        ),
        "reverse_dependency_allowed": False,
        "consumer_repository_identity_in_pssolver_docs": False,
        "consumer_objectives_or_experiments_in_pssolver": False,
    }
    assert "functional_execution_and_differentiability_capabilities" in (
        ownership["pssolver"]
    )
    assert "durable_checkpoint_conversion" in ownership["pssolver"]
    assert "adjoint_checkpoint_schedule" in ownership[
        "independent_control_consumer"
    ]
    assert "optimizers_line_searches_and_campaigns" in ownership[
        "independent_control_consumer"
    ]


def test_p90_binds_every_frozen_legacy_oracle_file_by_content() -> None:
    files = _record()["oracle_files"]

    assert len(files) == 13
    assert len({item["path"] for item in files}) == len(files)
    for item in files:
        path = ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]
        assert _git_blob(path) == item["git_blob"]


def test_p90_preserves_the_exact_legacy_public_surface() -> None:
    frozen = _record()["legacy_control"]

    assert frozen["status"] == (
        "frozen_compatibility_public_numerical_oracle"
    )
    assert frozen["new_features_allowed"] is False
    assert frozen["new_public_names_allowed"] is False
    assert list(legacy_control.__all__) == frozen["public_names"]
    readme = (ROOT / "pssolver" / "control" / "README.md").read_text(
        encoding="utf-8"
    )
    package_doc = legacy_control.__doc__ or ""
    assert "Status: frozen (2026-09-27)" in readme
    assert "independent consumer" in readme
    assert "Frozen compatibility surface" in package_doc


def test_p90_records_the_only_package_level_legacy_control_import() -> None:
    observed = []
    package = ROOT / "pssolver"
    for path in sorted(package.rglob("*.py")):
        if "control" in path.relative_to(package).parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "pssolver.control" in source or "from .control" in source:
            observed.append(path.relative_to(ROOT).as_posix())

    assert observed == ["pssolver/channel.py"]
    debt = _record()["legacy_control"]["only_production_import_debt"]
    assert debt["source"] == "pssolver.channel"
    assert debt["destination"] == "pssolver.control.active_force"


def test_p90_freezes_observation_replay_control_and_batch_semantics() -> None:
    contract = _record()["semantic_contract"]

    assert contract["observation_time"] == "input_state_under_current_control"
    assert contract["combined_and_separate_observation_bitwise_equal"] is True
    assert (
        contract["terminal_observation_includes_control_dependent_algebraic_fields"]
        is False
    )
    assert contract["functional_state_has_explicit_batch_axis"] is True
    assert contract["first_periodic_required_batch_size"] == 1
    assert contract["larger_batch_is_optional_declared_capability"] is True
    assert contract["functional_replay_same_execution_identity"] == "bitwise"
    assert contract["production_functional_comparison"] == (
        "strict_predeclared_tolerance"
    )
    assert contract["active_force_equation_term"] == "div(beta * alpha * Q)"
    assert contract["form_control_q_product_before_divergence"] is True
    assert contract["hidden_warm_state_allowed"] is False
    assert contract["diagnostic_cache_may_affect_future_step"] is False


def test_p90_freezes_periodic_first_slice_order_without_authorizing_p91() -> None:
    record = _record()
    sequence = record["phase_sequence"]
    decision = record["periodic_first_decision"]

    assert [item["id"] for item in sequence] == [
        "P9.0",
        "P9.1",
        "P9.2",
        "P9.3",
        "P9.4",
        "P9.5",
        "P9.6",
        "P9.7",
        "P9.8",
    ]
    assert sequence[0]["authorized"] is True
    assert sequence[0]["requires_h100"] is False
    assert all(item["authorized"] is False for item in sequence[1:])
    assert decision["first_application"] == (
        "periodic_complete_stress_beris_edwards"
    )
    assert decision["first_control_field"] == "activity"
    assert set(decision["p9_1_prerequisites"]) == {
        "public_positive_friction_periodic_application",
        "nz_one_and_z_invariant_contract",
        "explicit_2d_to_3d_q_embedding_and_projection",
    }
    assert all(value is False for value in record["non_claims"].values())


def test_future_verbatim_archive_lists_p90_without_rebuilding_the_pdf() -> None:
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "adr/0013-functional-runtime-and-external-control-ownership.md",
        "phase_8_final_closure.json",
        "phase_9_p90_control_contract_freeze.md",
        "phase_9_p90_control_contract_freeze.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
