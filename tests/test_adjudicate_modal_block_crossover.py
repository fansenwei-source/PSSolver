"""Fail-closed tests for P4.4 operator-contract evidence adjudication."""

from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

import pytest

import benchmarks.adjudicate_modal_block_crossover as adjudicator
from benchmarks.adjudicate_modal_block_crossover import (
    CANDIDATE,
    EXPECTED_COMMIT,
    EXPECTED_MODE_COUNTS,
    REFERENCE,
    EvidenceError,
    adjudicate_evidence,
    load_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION_RECORD = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p44_contract_equivalent_adjudication.json"
)


def _aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "trials": 3,
        "mean_milliseconds": statistics.fmean(values),
        "median_milliseconds": statistics.median(values),
        "sample_standard_deviation_milliseconds": statistics.stdev(values),
    }


def _row(mode_count: int) -> dict[str, object]:
    scale = mode_count / EXPECTED_MODE_COUNTS[0]
    reference = [1.00 * scale, 1.01 * scale, 0.99 * scale]
    candidate = [0.80 * value for value in reference]
    return {
        "mode_count": mode_count,
        "correctness": {
            "relative_l2": 1.5e-16,
            "linf": 4.0e-16,
            "residual_relative_l2": 1.4e-16,
            "finite": True,
            "reference_workspace_pointer_stable": True,
            "candidate_workspace_pointer_stable": True,
        },
        "trial_records": {
            REFERENCE: [
                {
                    "trial": index,
                    "order": 1 if index % 2 else 2,
                    "mean_milliseconds": value,
                }
                for index, value in enumerate(reference, start=1)
            ],
            CANDIDATE: [
                {
                    "trial": index,
                    "order": 2 if index % 2 else 1,
                    "mean_milliseconds": value,
                }
                for index, value in enumerate(candidate, start=1)
            ],
        },
        "aggregate": {
            REFERENCE: _aggregate(reference),
            CANDIDATE: _aggregate(candidate),
        },
        "candidate_over_reference_median_ratio": 0.8,
        "paired_candidate_over_reference_ratios": [0.8, 0.8, 0.8],
        "memory": {
            REFERENCE: {
                "starting_allocated_bytes": 100,
                "starting_reserved_bytes": 200,
                "peak_allocated_bytes": 200,
                "peak_reserved_bytes": 400,
                "incremental_peak_allocated_bytes": 100,
                "incremental_peak_reserved_bytes": 200,
            },
            CANDIDATE: {
                "starting_allocated_bytes": 100,
                "starting_reserved_bytes": 200,
                "peak_allocated_bytes": 160,
                "peak_reserved_bytes": 320,
                "incremental_peak_allocated_bytes": 60,
                "incremental_peak_reserved_bytes": 120,
            },
        },
    }


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": "p4_4_modal_block_crossover_profile",
        "scope": "evidence_only_no_production_selection",
        "config": {
            "mode_counts": list(EXPECTED_MODE_COUNTS),
            "device": "cuda",
            "dtype": "complex128",
            "alpha": 4.0,
            "warmup": 5,
            "repeats": 30,
            "trials": 3,
            "seed": 20260922,
            "maximum_acceptable_median_ratio": 1.05,
        },
        "environment": {
            "git_head": EXPECTED_COMMIT,
            "device": "cuda",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul_effective": False,
            "tf32_cudnn_effective": False,
        },
        "rows": [_row(mode_count) for mode_count in EXPECTED_MODE_COUNTS],
        "crossover": {
            "recommended_switch_at_mode_count": EXPECTED_MODE_COUNTS[0],
            "eligible_for_auto_policy_implementation": True,
            "production_default_changed": False,
            "p4_5_authorized": False,
        },
    }


def test_contract_equivalent_evidence_authorizes_p45_without_auto_policy():
    result = adjudicate_evidence(_payload(), input_sha256="a" * 64)
    assert result["classification"] == (
        "PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT"
    )
    assert result["contract_correction"] == {
        "authoritative_scope": "complete_operator_contract",
        "superseded_gate": (
            "reproduce_non_equivalent_bare_kernel_low_endpoint"
        ),
        "old_low_endpoint_direction_required": False,
        "reason": (
            "the earlier reference and the complete operator-contract "
            "reference measured different execution scopes"
        ),
        "auto_policy_required": False,
    }
    assert result["eligibility"] == {
        "p4_4_h100_qualified": True,
        "eligible_for_p4_5": True,
        "auto_policy_implemented": False,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
    assert len(result["rows"]) == 6


def test_machine_record_captures_formal_adjudication_and_p45_authority():
    record = json.loads(ADJUDICATION_RECORD.read_text(encoding="utf-8"))
    assert record["status"] == (
        "PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT"
    )
    assert record["input_evidence"]["job_id"] == 10837117
    assert record["contract_correction"] == {
        "authoritative_scope": "complete_operator_contract",
        "superseded_gate": (
            "reproduce_non_equivalent_bare_kernel_low_endpoint"
        ),
        "old_low_endpoint_direction_required": False,
        "auto_policy_required": False,
        "production_default_changed": False,
    }
    assert record["local_validation"] == {
        "adjudicator_tests_passed": 13,
        "phase_4_focused_tests_passed": 131,
        "complete_tests_passed": 1802,
        "subtests_passed": 8,
        "failures": 0,
        "archive_sources_verified": 62,
    }
    assert record["eligibility"] == {
        "contract_correction_authorized": True,
        "analysis_only_adjudication_authorized": False,
        "p4_4_h100_qualified": True,
        "eligible_for_p4_5": True,
        "auto_policy_implemented": False,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["environment"].__setitem__(
                "git_head", "0" * 40
            ),
            "git_head",
        ),
        (
            lambda payload: payload["config"].__setitem__("dtype", "complex64"),
            "dtype",
        ),
        (
            lambda payload: payload["rows"][0]["correctness"].__setitem__(
                "relative_l2", 2.0e-12
            ),
            "relative_l2",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__(
                "candidate_over_reference_median_ratio", 1.2
            ),
            "median_ratio",
        ),
        (
            lambda payload: payload["rows"][0]["memory"][
                CANDIDATE
            ].__setitem__("peak_allocated_bytes", 240),
            "allocated",
        ),
        (
            lambda payload: payload["rows"].pop(),
            "six registered mode counts",
        ),
        (
            lambda payload: payload["crossover"].__setitem__(
                "recommended_switch_at_mode_count", None
            ),
            "raw recommended switch",
        ),
    ],
)
def test_adjudicator_fails_closed_on_invalid_evidence(mutate, message):
    payload = copy.deepcopy(_payload())
    mutate(payload)
    with pytest.raises(EvidenceError, match=message):
        adjudicate_evidence(payload, input_sha256="b" * 64)


def test_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}\n')
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        load_evidence(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n')
    with pytest.raises(EvidenceError, match="non-finite JSON"):
        load_evidence(nonfinite)


def test_cli_writes_pass_report_without_mutating_input(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    original = json.dumps(_payload(), allow_nan=False, sort_keys=True) + "\n"
    input_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjudicate_modal_block_crossover",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    adjudicator.main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["classification"].startswith("PASS_P4_4")
    assert input_path.read_text(encoding="utf-8") == original


def test_cli_writes_failure_report_and_returns_nonzero(tmp_path, monkeypatch):
    payload = _payload()
    payload["environment"]["device_name"] = "not-an-H100"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "failure.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjudicate_modal_block_crossover",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        adjudicator.main()
    assert error.value.code == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["classification"] == (
        "FAIL_P4_4_CONTRACT_EQUIVALENT_ADJUDICATION"
    )
    assert result["eligibility"]["eligible_for_p4_5"] is False


def test_cli_refuses_to_overwrite_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    output_path.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjudicate_modal_block_crossover",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit):
        adjudicator.parse_args()
    assert output_path.read_text(encoding="utf-8") == "preserve\n"
