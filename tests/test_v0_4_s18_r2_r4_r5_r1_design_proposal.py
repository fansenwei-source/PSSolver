"""Bindings for the design-only S-18 R2/R4/R5/R1 proposal."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s18_r2_r4_r5_r1_design_proposal.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_s18_order_is_risk_increasing_and_batches_are_isolated():
    record = _read()
    assert record["recommended_batch_order"] == ["R2", "R4", "R5", "R1"]
    assert set(record["risk_rationale"]) == {"R2", "R4", "R5", "R1"}
    assert record["batch_policy"] == {
        "one_branch_per_batch": True,
        "one_independent_review_per_batch": True,
        "push_only_after_batch_pass_and_separate_authorization": True,
        "batch_parent_is_previous_accepted_dev_0_4_tip": True,
        "runtime_changes_in_this_design_commit": False,
    }


def test_s18_has_no_speculative_nonzero_tolerance():
    tolerance = _read()["tolerance_preregistration"]
    assert tolerance["nonzero_tolerances_registered_now"] == []
    fallback = tolerance["if_bitwise_equivalence_is_impossible"]
    assert fallback["action"] == ("stop before completing or integrating the affected batch")
    assert len(fallback["required_baseline_measurements"]) == 3
    assert len(fallback["required_registration_fields"]) == 5
    assert "explicit user confirmation" in fallback["authorization"]
    invariants = _read()["global_invariants"]
    assert invariants["default_numerical_comparison"] == "bitwise equality"
    assert invariants["no_existing_threshold_relaxation"] is True
    assert invariants["no_new_skip_or_xfail"] is True
    assert invariants["d_0013_memory_exception_extends_to_v0_4"] is False


def test_r2_freezes_released_formats_and_restart_behavior():
    r2 = _read()["batches"]["R2"]
    assert r2["bitwise_feasibility"] == "required_and_expected"
    paths = set(r2["existing_files_in_scope"])
    assert {
        "pssolver/io/checkpoint.py",
        "pssolver/functional/periodic_checkpoint.py",
        "pssolver/functional/channel_checkpoint.py",
        "pssolver/workflows/plane_checkpoint.py",
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/workflows/channel_checkpoint.py",
    } <= paths
    evidence = " ".join(r2["batch_specific_evidence"])
    assert "functional_checkpoint_releases_v1" in evidence
    assert "v0.3.0-tag-generated" in evidence
    assert "byte-tree comparison" in evidence
    assert "N-1" in evidence
    assert all(
        word in " ".join(r2["exact_requirements"])
        for word in ("bytes", "exception", "publication", "restart", "read-only")
    )


def test_r4_preserves_operation_order_and_progress_clocks():
    r4 = _read()["batches"]["R4"]
    assert r4["planned_internal_owner"].startswith("pssolver.integrators")
    requirements = " ".join(r4["exact_requirements"])
    for phrase in (
        "operation order",
        "in-place",
        "transform",
        "completed_steps",
        "callback timing",
        "restart state",
    ):
        assert phrase in requirements
    assert any("refresh intervals" in item for item in r4["batch_specific_evidence"])
    assert any("torch.compile" in item for item in r4["s19_gpu_scope"])


def test_r5_names_all_four_pressure_paths_and_keeps_hot_path_gates():
    r5 = _read()["batches"]["R5"]
    assert len(r5["four_paths"]) == 4
    paths = " ".join(r5["four_paths"])
    for phrase in (
        "_solve_pressure",
        "functional transpose",
        "implicit backward",
        "unrolled_channel_pressure_solve_oracle",
    ):
        assert phrase in paths
    requirements = " ".join(r5["exact_requirements"])
    for phrase in (
        "call order and counts",
        "host synchronization",
        "true residual",
        "gradients",
        "warm-start cache",
    ):
        assert phrase in requirements
    assert any("D-0013" in item for item in r5["s19_gpu_scope"])


def test_r1_covers_exactly_eight_stack_layers_and_uses_shadow_rollout():
    r1 = _read()["batches"]["R1"]
    layers = r1["eight_stack_layers_and_files"]
    assert len(layers) == 8
    assert [item["layer"] for item in layers] == [
        "capability declaration",
        "public simulation declaration",
        "lowering",
        "runtime binding",
        "public compiler dispatch",
        "application adapter",
        "runtime construction",
        "workflow exposure",
    ]
    assert r1["rollout"][0].startswith("construct a shadow registry")
    assert "switch one internal dispatch entry point" in r1["rollout"][2]
    requirements = " ".join(r1["exact_requirements"])
    assert "deterministic ordering" in requirements
    assert "runtime_identity" in requirements
    assert "configuration_identity" in requirements


def test_control_first_boundary_is_exact_and_research_stays_out_of_scope():
    control = _read()["control_first_contract"]
    assert control["functional_api_module"] == "pssolver.functional.api"
    assert control["functional_api_version"] == "1.1"
    assert control["pinned_constitutive_blobs"] == {
        "pssolver/models/active_nematics/beris_edwards.py": (
            "07f1f03f7f2f35e28e8ce629c1d73c5d59e4f5f3"
        ),
        "pssolver/models/active_nematics/q_tensor.py": ("5427df4402f31fc43ac409e1936cd31e58269966"),
    }
    assert "before the Provider change may be pushed" in control["pre_push_rule"]
    assert control["research_branch"] == {
        "branch": "research/3d-warmup",
        "provider": "v0.2.0",
        "included": False,
    }


def test_s18_design_binds_five_confirmed_choices_and_authorizes_no_remote_work():
    record = _read()
    assert record["classification"] == ("DESIGN_ONLY_D0018_CONFIRMED_AWAITING_R0054_REVIEW")
    governance = record["governance"]
    assert governance == {
        "confirmed_decisions": ["D-0011", "D-0014", "D-0017", "D-0018"],
        "confirmed_decision": "D-0018",
        "confirmed_decision_status": "confirmed",
        "confirmed_decision_sha256": (
            "38cfe71b63efcca6ef47a759d5955bd332266cf4a65cef6e61635c43bfd1a627"
        ),
        "confirmed_decision_sync_commit": ("4cf34051cd5d4f47801eaabf0bdc7cd8490d6367"),
        "implementation_requires_design_review_pass": True,
        "design_review_request": "R-0054",
        "superseded_preconfirmation_request": "R-0053",
    }
    assert [item["id"] for item in record["confirmed_choices"]] == [
        "Q-S18-01",
        "Q-S18-02",
        "Q-S18-03",
        "Q-S18-04",
        "Q-S18-05",
    ]
    assert all(item["selected_option"] == "A" for item in record["confirmed_choices"])
    assert record["implementation"] == {
        "started": False,
        "runtime_files_changed": [],
        "packaging_files_changed": [],
        "public_interfaces_changed": [],
        "numerical_output_changed": False,
        "checkpoint_schema_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
    }
    assert all(value is False for value in record["authorization"].values())
