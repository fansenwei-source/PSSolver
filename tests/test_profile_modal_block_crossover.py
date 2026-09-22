"""CPU contracts for the P4.4 modal-block crossover profiler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import benchmarks.profile_modal_block_crossover as crossover
from benchmarks.profile_modal_block_crossover import (
    CANDIDATE,
    REFERENCE,
    ModalBlockCrossoverConfig,
    run_crossover_profile,
    select_stable_suffix_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_RECORD = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p44_h100_recovery.json"
)


def _small_config(**updates) -> ModalBlockCrossoverConfig:
    values = {
        "mode_counts": (4, 8),
        "device": "cpu",
        "dtype": "complex128",
        "alpha": 4.0,
        "warmup": 0,
        "repeats": 1,
        "trials": 1,
        "seed": 2404,
        "maximum_acceptable_median_ratio": 1.05,
    }
    values.update(updates)
    return ModalBlockCrossoverConfig(**values)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"mode_counts": ()}, "mode_counts"),
        ({"mode_counts": (8, 4)}, "mode_counts"),
        ({"mode_counts": (4, 4)}, "mode_counts"),
        ({"mode_counts": (0, 4)}, "mode_counts"),
        ({"device": "meta"}, "device"),
        ({"dtype": "float64"}, "dtype"),
        ({"alpha": 0.0}, "alpha"),
        ({"warmup": -1}, "warmup"),
        ({"repeats": 0}, "repeats"),
        ({"trials": 0}, "trials"),
        ({"maximum_acceptable_median_ratio": 0.0}, "median ratio"),
    ],
)
def test_crossover_config_rejects_invalid_inputs(updates, message):
    with pytest.raises(ValueError, match=message):
        run_crossover_profile(_small_config(**updates))


def test_stable_suffix_threshold_requires_every_larger_case_to_pass():
    rows = [
        {"mode_count": 8, "candidate_over_reference_median_ratio": 1.4},
        {"mode_count": 16, "candidate_over_reference_median_ratio": 0.9},
        {"mode_count": 32, "candidate_over_reference_median_ratio": 1.01},
    ]
    assert select_stable_suffix_threshold(rows, maximum_ratio=1.05) == 16
    assert select_stable_suffix_threshold(rows, maximum_ratio=0.95) is None


def test_small_cpu_profile_is_finite_auditable_and_non_promoting():
    result = run_crossover_profile(_small_config())
    assert result["schema_version"] == 1
    assert result["identity"] == "p4_4_modal_block_crossover_profile"
    assert result["scope"] == "evidence_only_no_production_selection"
    assert result["environment"]["device"] == "cpu"
    assert result["variants"] == {
        REFERENCE: "torch.linalg.solve with shared input/workspace gates",
        CANDIDATE: "closed_form_2x2 with fixed P4.4 workspace",
    }
    assert [row["mode_count"] for row in result["rows"]] == [4, 8]
    for row in result["rows"]:
        assert row["correctness"]["finite"] is True
        assert row["correctness"]["relative_l2"] <= 1.0e-12
        assert row["correctness"]["linf"] <= 1.0e-12
        assert row["correctness"]["residual_relative_l2"] <= 1.0e-12
        assert row["aggregate"][REFERENCE]["trials"] == 1
        assert row["aggregate"][CANDIDATE]["trials"] == 1
        assert row["memory"] == {REFERENCE: None, CANDIDATE: None}
    assert result["crossover"]["production_default_changed"] is False
    assert result["crossover"]["p4_5_authorized"] is False


def test_cuda_request_fails_closed_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        run_crossover_profile(_small_config(device="cuda"))


def test_cli_writes_one_complete_json_artifact(tmp_path, monkeypatch):
    output = tmp_path / "crossover.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_modal_block_crossover",
            "--mode-counts",
            "4,8",
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--trials",
            "1",
            "--output",
            str(output),
        ],
    )
    crossover.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["identity"] == "p4_4_modal_block_crossover_profile"
    assert payload["config"]["mode_counts"] == [4, 8]
    assert payload["crossover"]["production_default_changed"] is False


def test_cli_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_modal_block_crossover",
            "--mode-counts",
            "4",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        crossover.parse_args()
    assert output.read_text(encoding="utf-8") == "preserve\n"


def test_recovery_record_preserves_failed_h100_evidence_and_blocks_p45():
    record = json.loads(RECOVERY_RECORD.read_text(encoding="utf-8"))
    assert record["status"] == "P4_4_CONTRACT_EQUIVALENT_ADJUDICATION_READY"
    assert record["h100_attempt"]["job_id"] == 10837110
    cases = record["h100_attempt"]["cases"]
    assert [case["mode_count"] for case in cases] == [131072, 4194304]
    assert cases[0]["performance_passed"] is False
    assert cases[1]["performance_passed"] is True
    assert record["crossover_attempt"]["job_id"] == 10837117
    assert record["crossover_attempt"][
        "all_paired_trials_favored_candidate"
    ] is True
    assert record["recovery_profiler"]["mode_counts"] == [
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
    ]
    assert record["eligibility"] == {
        "p4_4_h100_qualified": False,
        "crossover_h100_scan_authorized": False,
        "analysis_only_adjudication_authorized": True,
        "auto_policy_implemented": False,
        "p4_5_authorized": False,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
