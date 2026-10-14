from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc3_known_issues_matrix.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_matrix_binds_the_released_rc3_baseline_without_claiming_runtime_changes():
    record = _record()
    assert record["schema_version"] == 1
    assert record["baseline"] == {
        "release": "v0.2.0rc3",
        "commit": "5071d73a00e0d19be62ebd39edf9918818d1267a",
        "branch": "fix/v0.2.0rc4-audit",
        "matrix_changes_runtime": False,
        "released_tag_is_immutable": True,
    }
    assert record["release_boundary"] == {
        "rc3_declared_scope_qualified": True,
        "all_audit_items_closed": False,
        "rc4_correctness_batches_complete": True,
        "eligible_for_rc4_release_preparation": True,
        "eligible_for_final_v0_2_0": False,
        "rc4_post_release_verification_passed": False,
        "rc4_post_release_classification": "FAIL",
        "eligible_for_rc5_repair": True,
        "eligible_for_rc5_release": False,
        "new_features_authorized": False,
        "production_default_changed": False,
        "pssolver_control_change_required_for_this_matrix": False,
        "nematics3d_modified": False,
    }


def test_matrix_binds_both_external_audit_documents_by_digest():
    assert _record()["source_audits"] == [
        {
            "name": "PSSolver_v0_2_0rc1_bug_redundancy_audit_zh.pdf",
            "sha256": "a3a6edbc7c09ab09760230a9aaa22d0ffa552bae272fd48e1787123f54d9d4d1",
        },
        {
            "name": "PSSolver_v0_2_0rc2_fix_verification_zh.pdf",
            "sha256": "140986b3f6f20b0c0c874b21df02b037cf8ee1bb32e43dd11bc2d638cd9b8f76",
        },
        {
            "name": "PSSolver_v0_2_0rc4_post_release_verification.json",
            "sha256": "24f7e1a1129b72894c634071d8f272dfbd3f1a91017650427550e1f35f5da773",
            "classification": "FAIL_NEW_REGRESSION",
        },
        {
            "name": "PSSolver_v0_2_0rc4_post_release_verification_codex_draft.json",
            "sha256": "54e779aee7f622b78580aad1e03d26afea0eeaf6be8bcb07788bc8b53bb2e77b",
            "classification": "SUPERSEDED_PASS_WITH_DOCUMENTED_LIMITATIONS",
        },
        {
            "name": "PSSolver_v0_2_0rc4_post_release_verification_claude_review.json",
            "sha256": "e29501ab68088d18ba567058554406e5742cb5c923a8161d928c7e7b7862ac66",
            "classification": "FAIL",
        },
        {
            "name": "PSSolver_v0_2_0rc4_consolidated_report.json",
            "sha256": "db3e644bd97bc28ca046d0632aa2eae1b142ad6a208ad10b9dcb82cc42e6e6c6",
            "classification": "FAIL",
        },
    ]


def test_matrix_has_each_audit_identifier_exactly_once():
    items = _record()["items"]
    identifiers = [item["id"] for item in items]
    expected = (
        [f"B{index}" for index in range(1, 31)]
        + [f"A{index}" for index in range(1, 6)]
        + [f"N{index}" for index in range(1, 7)]
        + [f"R{index}" for index in range(1, 10)]
    )
    assert identifiers == expected
    assert len(set(identifiers)) == 50


def test_state_counts_match_the_machine_readable_summary():
    record = _record()
    counts: dict[str, int] = {}
    for item in record["items"]:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    normalized_counts = {
        "closed_verified": 28,
        "partial_residual": counts.get("partial_residual", 0),
        "regressed_after_closure": counts.get("regressed_after_closure", 0),
        "accepted_limitation": 3,
        "deferred_architecture": 9,
    }
    assert counts == {
        key: value for key, value in normalized_counts.items() if value
    }
    assert normalized_counts["partial_residual"] == 9
    assert normalized_counts["regressed_after_closure"] == 1
    assert record["summary"] == {
        "item_count": 50,
        **normalized_counts,
        "post_release_finding_count": 15,
        "first_implementation_batch": "rc4.1_plane_nyquist",
        "next_implementation_batch": "rc5_documentation",
    }


def test_release_blockers_and_closed_rc3_regressions_are_not_conflated():
    items = {item["id"]: item for item in _record()["items"]}
    for identifier in ("B1", "B2", "B3", "B4", "N1", "N2"):
        assert items[identifier]["state"] == "closed_verified"
    assert items["B5"]["state"] == "closed_verified"
    assert items["B5"]["disposition"] == "closed"
    assert items["B5"]["closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc412_h100_closure.json"
    )
    assert items["B10"]["state"] == "regressed_after_closure"
    assert items["B10"]["historical_closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc426_h100_closure.json"
    )
    for identifier in ("N3", "N6"):
        assert items[identifier]["state"] == "partial_residual"
        assert items[identifier]["historical_closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc426_h100_closure.json"
        )
    for identifier in ("B22", "N5"):
        assert items[identifier]["state"] == "closed_verified"
        assert items[identifier]["closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc426_h100_closure.json"
        )
    assert items["B8"]["state"] == "closed_verified"
    assert items["B8"]["physical_lifting_repaired"] is False
    assert items["N4"]["state"] == "closed_verified"
    assert items["B20"]["state"] == "partial_residual"
    for identifier in ("B8", "N4"):
        assert items[identifier]["closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc437_h100_closure.json"
        )
    assert items["B20"]["historical_closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc437_h100_closure.json"
    )
    for identifier in ("B25", "B28", "B29"):
        assert items[identifier]["state"] == "closed_verified"
        assert items[identifier]["disposition"] == "closed"
        assert items[identifier]["closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc447_h100_closure.json"
        )
    for identifier in ("B7", "B24"):
        assert items[identifier]["state"] == "partial_residual"
        assert items[identifier]["historical_closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
        )
    assert items["B27"]["state"] == "closed_verified"
    assert items["B27"]["closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
    )
    for identifier in ("B9", "B21"):
        assert items[identifier]["state"] == "closed_verified"
        assert items[identifier]["disposition"] == "closed"
        assert items[identifier]["closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
        )
    for identifier in ("B13", "B17"):
        assert items[identifier]["state"] == "partial_residual"
        assert items[identifier]["historical_closure_record"] == (
            "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
        )
    assert items["B15"]["state"] == "closed_verified"
    assert items["B15"]["closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
    )
    assert items["B11"]["state"] == "partial_residual"
    assert items["B11"]["historical_closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
    )
    assert items["B23"]["state"] == "partial_residual"
    assert items["B30"]["documentation_complete"] is False
    assert items["A3"]["target"] == "rc3"


def test_first_batch_is_only_plane_nyquist_and_has_non_regression_gates():
    record = _record()
    first = record["work_queue"][0]
    assert first["batch"] == "rc4.1_plane_nyquist"
    assert first["items"] == ["B5"]
    assert first["consumer_requalification"] is False
    assert set(first["required_gates"]) == {
        "manufactured_stokes_cpu",
        "odd_grid_byte_identity",
        "even_grid_storage_equivalence",
        "single_h100_non_regression",
    }


def test_architecture_work_is_deferred_until_correctness_batches_close():
    record = _record()
    architecture = record["work_queue"][-1]
    assert architecture["batch"] == "post_0_2_architecture"
    assert architecture["items"] == [f"R{index}" for index in range(1, 10)]
    assert all(
        item["state"] == "deferred_architecture"
        for item in record["items"]
        if item["id"].startswith("R")
    )


def test_post_release_findings_are_registered_without_changing_the_50_item_total():
    record = _record()
    findings = record["post_release_findings"]
    assert [item["id"] for item in findings] == [
        f"NEW-{index}" for index in range(1, 16)
    ]
    assert all(item["state"] == "open" for item in findings)
    assert len(record["items"]) == 50
    assert record["post_release"] == {
        "release": "v0.2.0rc4",
        "date": "2026-10-14",
        "classification": "FAIL",
        "decisive_item": "B10",
        "evidence_registration": (
            "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
        ),
        "historical_closure_records_rewritten": False,
        "runtime_changed_by_registration": False,
        "tag_or_release_mutated": False,
        "nematics3d_modified": False,
        "h100_submitted": False,
    }


def test_rc5_work_queue_matches_the_post_release_priority_order():
    queues = {item["batch"]: item for item in _record()["work_queue"]}
    assert queues["rc5_documentation"]["items"] == ["B30", "NEW-4", "NEW-5"]
    assert queues["rc5_must"]["items"] == [
        "B10", "NEW-1", "N3", "NEW-2", "NEW-3", "B11"
    ]
    assert queues["rc5_should"]["items"] == [
        "B7",
        "B13",
        "B17",
        "B20",
        "B23",
        "B24",
        "N6",
        "NEW-7",
        "NEW-8",
        "NEW-9",
        "NEW-10",
        "NEW-11",
        "NEW-12",
        "NEW-13",
        "NEW-14",
    ]
    assert queues["rc5_optional"]["items"] == [
        "B14", "B27", "B28", "N4", "NEW-15"
    ]
    assert queues["rc5_release_process"]["items"] == ["NEW-6"]
    assert queues["rc5_qualification"]["items"] == ["G-1"]


def test_rc450_progress_records_rc44_closure_and_rc45_planning_boundary():
    progress = _record()["progress"]
    assert progress == {
        "completed_batches": [
            "rc4.1_plane_nyquist",
            "rc4.2_checkpoint_identity",
            "rc4.3_lifting_correctness",
            "rc4.4_pressure_contract",
            "rc4.5_cross_runtime_robustness",
        ],
        "next_batch": "post_0_2_architecture",
        "b5_complete": True,
        "b5_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc412_h100_closure.json"
        ),
        "eligible_for_rc4_2_planning": True,
        "rc4_2_planning_complete": True,
        "rc4_2_plan_record": (
            "notes/PSSolver_v0_2_0rc4_rc420_checkpoint_identity_plan.json"
        ),
        "rc4_2_complete": True,
        "rc4_2_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc426_h100_closure.json"
        ),
        "eligible_for_rc4_3_planning": True,
        "rc4_3_planning_complete": True,
        "rc4_3_plan_record": (
            "notes/PSSolver_v0_2_0rc4_rc430_lifting_correctness_plan.json"
        ),
        "rc4_3_1_analysis_authorized": True,
        "rc4_3_1_analysis_complete": True,
        "rc4_3_1_classification": "PUBLIC_CAPABILITY_RESTRICTION_REQUIRED",
        "rc4_3_1_analysis_record": (
            "notes/PSSolver_v0_2_0rc4_rc431_lifting_correctness_analysis.json"
        ),
        "rc4_3_1_analysis_record_sha256": (
            "1157b08b5328996876307fb0333aa28e85932d9d09e9ed0f16f08cc7a3a62ef1"
        ),
        "eligible_for_rc4_3_2_planning": True,
        "rc4_3_2_contract_authorized": True,
        "rc4_3_2_contract_complete": True,
        "rc4_3_2_classification": (
            "PASS_RC4_3_2_PUBLIC_LIFTING_CAPABILITY_RESTRICTION_CONTRACT"
        ),
        "rc4_3_2_contract_record": (
            "notes/PSSolver_v0_2_0rc4_rc432_lifting_capability_contract.json"
        ),
        "rc4_3_2_contract_record_sha256": (
            "5e6b64626f1927eb9846f62b3fbe31243d96131bbf2d3d0e36aeb254018d2741"
        ),
        "eligible_for_rc4_3_3_implementation": True,
        "rc4_3_3_implementation_authorized": True,
        "rc4_3_3_implementation_complete": True,
        "rc4_3_3_classification": (
            "PASS_RC4_3_3_PUBLIC_LIFTING_CAPABILITY_RESTRICTION"
        ),
        "rc4_3_3_record": (
            "notes/PSSolver_v0_2_0rc4_rc433_lifting_capability_restriction.json"
        ),
        "eligible_for_rc4_3_4_implementation": True,
        "rc4_3_implementation_authorized": True,
        "rc4_3_4_implementation_complete": True,
        "rc4_3_4_classification": (
            "PASS_RC4_3_4_EXPLICIT_FRESH_CONDITIONING_CPU_CANDIDATE"
        ),
        "rc4_3_4_record": (
            "notes/PSSolver_v0_2_0rc4_rc434_fresh_conditioning.json"
        ),
        "rc4_3_5_implementation_complete": True,
        "rc4_3_5_classification": (
            "PASS_RC4_3_5_LIFTING_RUNTIME_CHECKPOINT_INTEGRATION_CPU_CANDIDATE"
        ),
        "rc4_3_5_record": (
            "notes/PSSolver_v0_2_0rc4_rc435_runtime_checkpoint_integration.json"
        ),
        "rc4_3_6_installed_cpu_complete": True,
        "rc4_3_6_classification": (
            "PASS_RC4_3_6_INSTALLED_WHEEL_LIFTING_CPU_CLOSURE"
        ),
        "rc4_3_6_record": (
            "notes/PSSolver_v0_2_0rc4_rc436_installed_cpu_closure.json"
        ),
        "rc4_3_7_h100_complete": True,
        "rc4_3_7_classification": (
            "PASS_RC4_3_7_INTERNAL_LIFTING_SINGLE_H100_CLOSURE"
        ),
        "rc4_3_7_record": (
            "notes/PSSolver_v0_2_0rc4_rc437_h100_closure.json"
        ),
        "rc4_3_complete": True,
        "rc4_3_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc437_h100_closure.json"
        ),
        "eligible_for_rc4_4_planning": True,
        "rc4_4_planning_complete": True,
        "rc4_4_plan_classification": (
            "READY_RC4_4_1_PRESSURE_CONTRACT_ADJUDICATION_NOT_EXECUTED"
        ),
        "rc4_4_plan_record": (
            "notes/PSSolver_v0_2_0rc4_rc440_pressure_contract_plan.json"
        ),
        "rc4_4_1_analysis_authorized": True,
        "rc4_4_1_analysis_complete": True,
        "rc4_4_1_classification": (
            "PASS_RC4_4_1_ACTIVE_MODE_RMS_AND_SCOPED_CAPABILITY_CONTRACT"
        ),
        "rc4_4_1_analysis_record": (
            "notes/PSSolver_v0_2_0rc4_rc441_pressure_adjudication.json"
        ),
        "eligible_for_rc4_4_2": True,
        "rc4_4_2_contract_authorized": True,
        "rc4_4_2_contract_complete": True,
        "rc4_4_2_classification": (
            "PASS_RC4_4_2_PUBLIC_CHANNEL_PRESSURE_CONTRACT"
        ),
        "rc4_4_2_contract_record": (
            "notes/PSSolver_v0_2_0rc4_rc442_pressure_contract.json"
        ),
        "eligible_for_rc4_4_3_implementation": True,
        "rc4_4_3_implementation_authorized": True,
        "rc4_4_implementation_authorized": True,
        "rc4_4_3_implementation_complete": True,
        "rc4_4_4_connection_complete": True,
        "rc4_4_5_installed_cpu_complete": True,
        "rc4_4_7_h100_complete": True,
        "rc4_4_7_classification": (
            "PASS_RC4_4_7_CHANNEL_PRESSURE_SINGLE_H100_CUMULATIVE_CLOSURE"
        ),
        "rc4_4_complete": True,
        "rc4_4_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc447_h100_closure.json"
        ),
        "rc4_4_closure_record_sha256": (
            "ee6c81e94e67ba38d280365adc211f8e32ec933fcccaf5474afb511d1b6c0482"
        ),
        "eligible_for_rc4_5_planning": True,
        "rc4_5_planning_complete": True,
        "rc4_5_plan_classification": (
            "READY_RC4_5_1_CROSS_RUNTIME_RESIDUAL_ADJUDICATION_NOT_EXECUTED"
        ),
        "rc4_5_plan_record": (
            "notes/PSSolver_v0_2_0rc4_rc450_cross_runtime_robustness_plan.json"
        ),
        "rc4_5_plan_record_sha256": (
            "2e0c122fbf7aa269c8746195c03b2d3f431e2f699c8536dc050b534266ad7c8d"
        ),
        "rc4_5_1_analysis_authorized": True,
        "rc4_5_1_analysis_complete": True,
        "rc4_5_1_classification": (
            "RC4_5_1_RESIDUAL_ADJUDICATION_COMPLETE"
        ),
        "rc4_5_1_analysis_record": (
            "notes/PSSolver_v0_2_0rc4_rc451_cross_runtime_adjudication.json"
        ),
        "rc4_5_1_analysis_record_sha256": (
            "0ab513797736922cdda66bfb9c87b1c070323475e4e5b120cedfe6955aa0568d"
        ),
        "eligible_for_rc4_5_2_contract": True,
        "rc4_5_2_contract_authorized": True,
        "rc4_5_2_contract_complete": True,
        "rc4_5_2_classification": (
            "PASS_RC4_5_2_PUBLIC_CROSS_RUNTIME_ROBUSTNESS_CONTRACT"
        ),
        "rc4_5_2_contract_record": (
            "notes/PSSolver_v0_2_0rc4_rc452_public_robustness_contract.json"
        ),
        "rc4_5_2_contract_record_sha256": (
            "ca8dbcf7b1b2d4144f28883f6a420331dbaf9af712d452a9d26551affaa9cdaf"
        ),
        "eligible_for_rc4_5_3_implementation": True,
        "rc4_5_3_implementation_authorized": True,
        "rc4_5_3_complete": True,
        "rc4_5_3_classification": (
            "PASS_RC4_5_3_CHECKPOINT_TRANSACTIONALITY_CPU_CANDIDATE"
        ),
        "rc4_5_3_record": (
            "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
        ),
        "rc4_5_3_record_sha256": (
            "4716b47e589dac8f13b56458ace00179e4fd0815042f81f2d034e6547edb03b4"
        ),
        "eligible_for_rc4_5_4_implementation": True,
        "rc4_5_4_implementation_authorized": True,
        "rc4_5_4_complete": True,
        "rc4_5_4_classification": (
            "PASS_RC4_5_4_BOUNDARY_SEMANTICS_CPU_CANDIDATE"
        ),
        "rc4_5_4_record": (
            "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
        ),
        "rc4_5_4_record_sha256": (
            "7e8a7ac578697d8fd06143e4b437c77a6ab576dd2d5b03d5a1f7c87c3deff8a8"
        ),
        "eligible_for_rc4_5_5_implementation": True,
        "rc4_5_5_implementation_authorized": True,
        "rc4_5_5_complete": True,
        "rc4_5_5_classification": (
            "PASS_RC4_5_5_PUBLIC_VALIDATION_OBSERVABILITY_CPU_CANDIDATE"
        ),
        "rc4_5_5_record": (
            "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
        ),
        "rc4_5_5_record_sha256": hashlib.sha256(
            (
                ROOT
                / "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
            ).read_bytes()
        ).hexdigest(),
        "eligible_for_rc4_5_6_implementation": True,
        "rc4_5_6_implementation_authorized": True,
        "rc4_5_6_complete": True,
        "rc4_5_6_classification": (
            "PASS_RC4_5_6_SCOPED_TF32_CPU_CANDIDATE"
        ),
        "rc4_5_6_record": (
            "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
        ),
        "rc4_5_6_record_sha256": hashlib.sha256(
            (
                ROOT
                / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
            ).read_bytes()
        ).hexdigest(),
        "eligible_for_rc4_5_7_installed_cpu_closure": True,
        "rc4_5_7_executed": True,
        "rc4_5_7_complete": True,
        "rc4_5_7_classification": (
            "PASS_RC4_5_7_INSTALLED_WHEEL_CPU_CONSUMER_CLOSURE_"
            "WITH_TF32_IDENTITY_MIGRATION"
        ),
        "rc4_5_7_record": (
            "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
        ),
        "rc4_5_7_record_sha256": hashlib.sha256(
            (
                ROOT
                / "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
            ).read_bytes()
        ).hexdigest(),
        "rc4_5_7_initial_record": (
            "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
        ),
        "rc4_5_7_initial_record_sha256": hashlib.sha256(
            (
                ROOT
                / "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
            ).read_bytes()
        ).hexdigest(),
        "rc4_5_7_consumer_migration_required": False,
        "rc4_5_7_consumer_migration_complete": True,
        "eligible_for_rc4_5_8_h100_closure": True,
        "rc4_5_8_h100_complete": True,
        "rc4_5_8_classification": (
            "PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE_"
            "WITH_COUNTER_AND_RESERVED_MEMORY_REDUCER_RECOVERY"
        ),
        "rc4_5_8_record": (
            "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
        ),
        "rc4_5_8_record_sha256": (
            hashlib.sha256(
                (
                    ROOT
                    / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
                ).read_bytes()
            ).hexdigest()
        ),
        "rc4_5_complete": True,
        "rc4_5_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
        ),
        "rc4_complete": True,
        "eligible_for_rc4_release_preparation": True,
        "rc4_5_implementation_authorized": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
    analysis = ROOT / progress["rc4_3_1_analysis_record"]
    assert hashlib.sha256(analysis.read_bytes()).hexdigest() == progress[
        "rc4_3_1_analysis_record_sha256"
    ]
    contract = ROOT / progress["rc4_3_2_contract_record"]
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == progress[
        "rc4_3_2_contract_record_sha256"
    ]
    assert (ROOT / progress["rc4_3_3_record"]).is_file()
    for stage in ("rc4_3_4", "rc4_3_5", "rc4_3_6", "rc4_3_7"):
        assert (ROOT / progress[f"{stage}_record"]).is_file()
    assert (ROOT / progress["rc4_3_closure_record"]).is_file()
    assert (ROOT / progress["rc4_4_1_analysis_record"]).is_file()
    assert (ROOT / progress["rc4_4_2_contract_record"]).is_file()
    for stage in (
        "rc4_4_closure",
        "rc4_5_plan",
        "rc4_5_1_analysis",
        "rc4_5_2_contract",
        "rc4_5_3",
        "rc4_5_4",
        "rc4_5_5",
        "rc4_5_6",
        "rc4_5_7",
        "rc4_5_8",
    ):
        path = ROOT / progress[f"{stage}_record"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == progress[
            f"{stage}_record_sha256"
        ]
