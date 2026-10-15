"""Contracts for the rc5 G-1 HPCC pre-submission evidence record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_hpcc_preflight.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_hpcc_preflight_stops_before_plan_freeze_or_h100_submission():
    record = _record()
    assert record["classification"] == (
        "STOPPED_PRE_STAGE_PLAN_MISSING_VERSIONED_PRODUCERS"
    )
    assert record["qualification_complete"] is False
    assert record["stage_plan_frozen"] is False
    assert record["h100_submission_authorized"] is False
    assert record["h100_job_submitted"] is False
    assert record["formal_h100_submission_count"] == 0
    assert record["automatic_retry"] is False
    assert record["hpcc"]["planned_h100_output_root_absent"] is True


def test_hpcc_preflight_binds_the_exact_archives_and_wheels():
    record = _record()
    assert record["support_source"] == {
        "branch": "fix/v0.2.0rc5",
        "commit": "3443f923b942e586a221241967823162a302d8ab",
        "archive_sha256": (
            "87cd60a0562b1a3096194a6ab430a509db8d3d7e4240ee8b4ed6edc6e81e7e19"
        ),
        "stage_runner_sha256": (
            "a633e183a3e4558952fb465712e2fe72a9ddf1d13eba582d7522ef5bfff8b9a5"
        ),
        "slurm_template_sha256": (
            "31f99a4906a39bd0361ef3ab3001d76b3507b19a5f6331c92a83019b7915481b"
        ),
    }
    artifacts = record["artifacts"]
    assert artifacts["provider_baseline_wheel_sha256"] == (
        "1bf31f324180dac58777a23785fb4e1424d3cc589099cf8c6328e2f6c74d8cc5"
    )
    assert artifacts["provider_candidate_wheel_sha256"] == (
        "3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f"
    )
    assert artifacts["consumer_wheel_sha256"] == (
        "3f5f359b0c7a7112b66245009a7e67e4479cab0ab6df630428c751dc26848745"
    )
    assert artifacts["remote_sha256_verification"] == "PASS"


def test_hpcc_preflight_records_three_clean_installed_wheel_environments():
    record = _record()
    venvs = record["installed_wheel_venvs"]
    assert venvs["baseline"]["pssolver_version"] == "0.2.0rc4"
    assert venvs["candidate"]["pssolver_version"] == "0.2.0rc5"
    assert venvs["consumer"]["pssolver_version"] == "0.2.0rc5"
    assert venvs["consumer"]["pssolver_control_version"] == "0.0.1.dev0"
    assert all(value["pip_check"] == "PASS" for value in venvs.values())
    assert record["support_isolation"][
        "provider_support_contains_pssolver_source_tree"
    ] is False
    assert record["support_isolation"]["source_shadow_import"] is False


def test_hpcc_preflight_names_the_exact_missing_producer_boundary():
    record = _record()
    assert record["missing_versioned_stage_producers"] == [
        "environment",
        "cuda_only",
        "checkpoints",
        "numerical",
        "consumer",
    ]
    assert record["existing_versioned_stage_producers"] == [
        "tf32_scope",
        "profiles",
        "assemble",
        "analyze",
    ]
    assert record["scope"] == {
        "pssolver_runtime_modified": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "scientific_threshold_changed": False,
    }
