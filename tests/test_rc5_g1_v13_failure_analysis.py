"""CPU-only contracts for the rc5 G-1 v13 failure diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc5_g1_v13_failure import (
    CLASSIFICATION,
    V10_CLASSIFICATION,
    V11_CLASSIFICATION,
    V12_CLASSIFICATION,
    V13_CLASSIFICATION,
    adjudicate,
    main,
)


CELLS = (
    "plane_compiled_v2/plane_R128",
    "channel_compiled_v2/channel_R128x20x20",
    "periodic_functional/periodic_R128",
    "channel_functional/channel_R512x40x40",
)
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(medians: list[float]) -> list[dict[str, object]]:
    result = []
    for trial, median in enumerate(medians, start=1):
        ratios = [median - 0.002, median - 0.001, median, median, median + 0.001, median + 0.002]
        result.append(
            {
                "trial": trial,
                "execution_order": (
                    ["baseline", "candidate"]
                    if trial % 2
                    else ["candidate", "baseline"]
                ),
                "baseline_cv_at_or_below_reference": True,
                "candidate_cv_at_or_below_reference": True,
                "fixed_block_ratios": ratios,
                "median_fixed_block_ratio": median,
            }
        )
    return result


def _group(cell: str, *, failed: bool, periodic_v13: bool = False) -> dict[str, object]:
    family, grid = cell.split("/")
    gates = {
        "aggregate_all_sample_mean": True,
        "median_of_profile_means": True,
        "median_of_all_fixed_block_ratios": True,
        "every_trial_median_fixed_block_ratio": not failed,
        "peak_allocated": True,
        "peak_reserved": True,
    }
    if periodic_v13:
        pairs = _pairs([0.998, 1.002, 1.063, 0.958, 0.934, 1.009])
        pairs[2]["fixed_block_ratios"] = [1.066, 1.064, 1.063, 1.068, 0.878, 0.995]
        pairs[2]["median_fixed_block_ratio"] = 1.0635
        pairs[4]["fixed_block_ratios"] = [0.933, 0.928, 0.934, 0.935, 0.936, 0.936]
        pairs[4]["median_fixed_block_ratio"] = 0.9345
    else:
        pairs = _pairs([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return {
        "family": family,
        "grid": grid,
        "passed": not failed,
        "gates": gates,
        "metrics": {
            "maximum_trial_median_fixed_block_ratio": (
                1.0635 if failed else 1.02
            )
        },
        "pairs": pairs,
    }


def _evidence() -> tuple[dict[str, object], ...]:
    v10 = {
        "classification": V10_CLASSIFICATION,
        "analysis_only": True,
        "proposed_contract_passed": False,
        "thresholds": {"every_trial_median_fixed_block_ratio_max": 1.05},
        "groups": [
            _group(cell, failed=(cell == CELLS[1]))
            for cell in CELLS
        ],
    }
    v13 = {
        "classification": V13_CLASSIFICATION,
        "analysis_only": False,
        "proposed_contract_passed": False,
        "formal_requalification_performed": True,
        "thresholds": {"every_trial_median_fixed_block_ratio_max": 1.05},
        "groups": [
            _group(
                cell,
                failed=(cell == CELLS[2]),
                periodic_v13=(cell == CELLS[2]),
            )
            for cell in CELLS
        ],
    }
    v11 = {"classification": V11_CLASSIFICATION, "analysis_only": True}
    v12 = {
        "classification": V12_CLASSIFICATION,
        "analysis_only": True,
        "adjudication": {
            "position_drifting_timing_variability_observed": True,
            "single_root_cause_resolved": False,
        },
    }
    return v10, v13, v11, v12


def test_failure_cell_migration_is_diagnostic_only_and_preserves_formal_fail():
    result = adjudicate(*_evidence())
    assert result["classification"] == CLASSIFICATION
    assert result["analysis_only"] is True
    assert result["formal_v13_classification_preserved"] == V13_CLASSIFICATION
    assert result["formal_v13_pass_granted"] is False
    assert result["qualification_complete"] is False
    assert result["g1_complete"] is False
    assert result["persistent_provider_regression_supported_by_current_evidence"] is False
    assert result["paired_block_contract_repeatability_concern_supported"] is True
    assert result["cross_run"]["v10_failed_cells"] == [CELLS[1]]
    assert result["cross_run"]["v13_failed_cells"] == [CELLS[2]]
    assert result["v13_periodic"]["failing_trials"] == [3]
    assert result["v13_periodic"]["trial3_blocks_above_threshold"] == 4
    assert result["v13_periodic"]["trial5_candidate_faster_in_every_block"] is True
    assert result["thresholds_changed"] is False
    assert result["samples_deleted"] is False
    assert result["trials_replaced"] is False


def test_same_failed_cell_or_persistent_gate_failure_is_rejected():
    v10, v13, v11, v12 = _evidence()
    v10["groups"] = [
        _group(cell, failed=(cell == CELLS[2]), periodic_v13=(cell == CELLS[2]))
        for cell in CELLS
    ]
    with pytest.raises(ValueError, match="did not migrate"):
        adjudicate(v10, v13, v11, v12)

    v10, v13, v11, v12 = _evidence()
    v13["groups"][2]["gates"]["aggregate_all_sample_mean"] = False
    with pytest.raises(ValueError, match="isolated to the maximum-trial gate"):
        adjudicate(v10, v13, v11, v12)


def test_tampered_periodic_pattern_or_prior_classification_is_rejected():
    v10, v13, v11, v12 = _evidence()
    v13["groups"][2]["pairs"][4]["fixed_block_ratios"] = [0.951] * 6
    v13["groups"][2]["pairs"][4]["median_fixed_block_ratio"] = 0.951
    with pytest.raises(ValueError, match="reverse-sign pattern differs"):
        adjudicate(v10, v13, v11, v12)

    v10, v13, v11, v12 = _evidence()
    v12["classification"] = "UNKNOWN"
    with pytest.raises(ValueError, match="v12 classification differs"):
        adjudicate(v10, v13, v11, v12)


def test_cli_writes_atomically_and_refuses_overwrite(tmp_path: Path):
    values = _evidence()
    paths = []
    for name, value in zip(("v10", "v13", "v11", "v12"), values, strict=True):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "analysis.json"
    arguments = [
        "--v10-analysis",
        str(paths[0]),
        "--v13-analysis",
        str(paths[1]),
        "--v11-analysis",
        str(paths[2]),
        "--v12-analysis",
        str(paths[3]),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == CLASSIFICATION
    assert set(payload["source_evidence"]) == {
        "v10_paired_block_replay",
        "v13_formal_analysis",
        "v11_timing_stability",
        "v12_attribution",
    }
    assert not list(tmp_path.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError):
        main(arguments)


def test_repository_record_preserves_the_formal_fail_and_binds_the_analyzer():
    record = json.loads(
        (
            ROOT
            / "notes"
            / "PSSolver_v0_2_0rc5_g1_v13_failure_analysis.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == CLASSIFICATION
    assert record["formal_v13"] == {
        "job_id": "10886110",
        "classification": V13_CLASSIFICATION,
        "classification_preserved": True,
        "pass_granted": False,
        "qualification_complete": False,
        "g1_complete": False,
        "profile_count": 48,
        "progress_record_count": 48,
        "failed_cell": CELLS[2],
        "failed_trial": 3,
        "maximum_trial_median_fixed_block_ratio": 1.0633863358803155,
        "threshold": 1.05,
    }
    diagnosis = record["cross_run_diagnosis"]
    assert diagnosis["v10_failed_cell"] == CELLS[1]
    assert diagnosis["v13_failed_cell"] == CELLS[2]
    assert diagnosis["failed_cell_migrated"] is True
    assert diagnosis["persistent_provider_regression_supported_by_current_evidence"] is False
    assert diagnosis["paired_block_contract_repeatability_concern_supported"] is True
    hpcc = record["hpcc_analysis"]
    assert hpcc["formal_h100_submission_count"] == 0
    assert hpcc["slurm_job_submitted"] is False
    assert hpcc["old_archive_checksum_results"] == {
        "v10": "41/41 PASS",
        "v11": "8/8 PASS",
        "v12": "83/83 PASS",
        "v13": "1803/1803 PASS",
    }
    assert hpcc["archive_checksum_result"] == "12/12 PASS"
    analyzer = ROOT / "benchmarks" / "analyze_rc5_g1_v13_failure.py"
    assert record["source_sha256"][str(analyzer.relative_to(ROOT))] == _sha256(analyzer)
    assert record["scope"] == {
        "analysis_only": True,
        "thresholds_changed": False,
        "samples_deleted": False,
        "trials_replaced": False,
        "formal_v13_reclassified": False,
        "new_h100_job_submitted": False,
        "production_solver_modified": False,
        "production_runtime_modified": False,
        "production_default_changed": False,
        "qualification_complete": False,
        "g1_complete": False,
    }
