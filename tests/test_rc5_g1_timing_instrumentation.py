"""CPU-only contracts for the rc5 G-1 timing instrumentation harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.plan_rc5_g1_timing_instrumentation import (
    FAMILIES,
    build_plan,
)
from benchmarks.run_rc5_g1_checkpoint_tmp_preflight import (
    SCHEMA as CHECKPOINT_TMP_PREFLIGHT_SCHEMA,
)
from benchmarks.run_rc5_g1_checkpoint_tmp_preflight import (
    _atomic_json as checkpoint_preflight_atomic_json,
)


def test_plan_freezes_analysis_only_scope_and_formal_boundaries():
    plan = build_plan()

    assert plan["classification"] == "READY_FOR_ANALYSIS_ONLY_INSTRUMENTATION"
    assert plan["families"] == list(FAMILIES)
    assert plan["profile_count"] == 12
    assert [row["sequence"] for row in plan["profile_matrix"]] == list(range(1, 13))
    assert {row["provider_role"] for row in plan["profile_matrix"]} == {"candidate"}
    assert all(row["timed_steps"] == 60 for row in plan["profile_matrix"])
    assert all(row["device"] == "cuda" for row in plan["profile_matrix"])
    assert {
        family: sum(row["family"] == family for row in plan["profile_matrix"])
        for family in FAMILIES
    } == {family: 3 for family in FAMILIES}

    boundaries = plan["frozen_boundaries"]
    assert boundaries["formal_raw_sample_count"] == 60
    assert boundaries["formal_cv_threshold"] == 0.05
    assert boundaries["formal_threshold_changed"] is False
    assert boundaries["outlier_removal_allowed"] is False
    assert boundaries["trial_replacement_allowed"] is False
    assert boundaries["formal_v10_reclassification_allowed"] is False

    authorization = plan["authorization"]
    assert authorization == {
        "implementation_and_cpu_tests_authorized": True,
        "push_authorized": False,
        "h100_submission_authorized": False,
        "automatic_retry_authorized": False,
        "production_default_changed": False,
    }


def test_checkpoint_tmp_preflight_helper_is_versioned_verbatim_and_atomic(
    tmp_path,
):
    root = Path(__file__).resolve().parents[1]
    helper = root / "benchmarks" / "run_rc5_g1_checkpoint_tmp_preflight.py"

    assert CHECKPOINT_TMP_PREFLIGHT_SCHEMA == ("pssolver.rc5.g1.checkpoint_tmp_preflight.v1")
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
        "ec6b4d2c70f5b4454754120b0f87a8771878ec50787a5590e23759bcc23cf142"
    )
    output = tmp_path / "preflight.json"
    checkpoint_preflight_atomic_json(
        output,
        {"schema": CHECKPOINT_TMP_PREFLIGHT_SCHEMA, "passed": True},
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passed": True,
        "schema": CHECKPOINT_TMP_PREFLIGHT_SCHEMA,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        checkpoint_preflight_atomic_json(output, {"passed": False})


def test_implementation_record_binds_sources_without_authorizing_h100():
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "notes" / "PSSolver_v0_2_0rc5_g1_timing_instrumentation_v12.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["classification"] == ("READY_FOR_ANALYSIS_ONLY_TIMING_INSTRUMENTATION")
    assert record["base_commit"] == ("2dabdf3439c2d4d602efc5f2a16ba0c7f1491803")
    assert record["scope"]["formal_requalification_performed"] is False
    assert record["scope"]["h100_execution_performed"] is False
    assert record["authorization"]["push_authorized"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["automatic_retry_authorized"] is False
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
