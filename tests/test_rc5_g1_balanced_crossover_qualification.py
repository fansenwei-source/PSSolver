"""Fail-closed CPU contracts for formal G-1 balanced-crossover support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.analyze_rc5_g1_balanced_crossover_qualification as formal_analyzer
from benchmarks.plan_rc5_g1_balanced_crossover_qualification import (
    CLASSIFICATION,
    SCHEMA,
    build_plan,
    main as plan_main,
)
from benchmarks.run_rc5_g1_balanced_crossover_profile_matrix import (
    AUTHORIZATION_SCHEMA,
    BOUND_SOURCE_FILES,
    PROGRESS_SCHEMA,
    SUPPORT_FILES,
    build_command_plan,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_plan_adopts_estimator_but_does_not_authorize_push_or_h100():
    plan = build_plan()
    assert plan["schema"] == SCHEMA
    assert plan["classification"] == CLASSIFICATION
    assert plan["profile_count"] == 48
    assert plan["adoption"]["v10_v13_frozen_replay_passed"] is True
    assert plan["adoption"]["formal_v10_classification_preserved"] is True
    assert plan["adoption"]["formal_v13_classification_preserved"] is True
    assert plan["adoption"]["formal_v13_pass_granted"] is False
    authorization = plan["authorization"]
    assert authorization["formal_contract_adoption_authorized"] is True
    assert authorization["push_authorized"] is False
    assert authorization["h100_submission_authorized"] is False
    assert authorization["maximum_h100_submissions_without_new_authorization"] == 0
    assert authorization["automatic_retry_authorized"] is False


def test_formal_plan_cli_is_atomic_and_refuses_overwrite(tmp_path):
    output = tmp_path / "plan.json"
    assert plan_main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    with pytest.raises(FileExistsError):
        plan_main(["--output", str(output)])


def test_formal_analyzer_uses_adopted_estimator_not_superseded_maximum(monkeypatch):
    monkeypatch.setattr(
        formal_analyzer,
        "validate_paired_block_evidence",
        lambda *args, **kwargs: {
            "classification": "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE",
            "proposed_contract_passed": False,
            "profile_count": 48,
            "execution_sequence_validated": True,
        },
    )
    monkeypatch.setattr(
        formal_analyzer,
        "_balanced_groups",
        lambda *args, **kwargs: [
            {"family": f"family-{index}", "grid": "grid", "passed": True}
            for index in range(4)
        ],
    )
    result = formal_analyzer.analyze(Path("profiles"), Path("progress"))
    assert result["classification"] == formal_analyzer.PASS_CLASSIFICATION
    assert result["analysis_only"] is False
    assert result["formal_requalification_performed"] is True
    assert result["superseded_maximum_trial_estimator_diagnostic"] == {
        "classification": "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE",
        "passed": False,
        "used_for_formal_adjudication": False,
        "reason": "the maximum-single-trial estimator was superseded by the adopted v14 contract",
    }
    assert result["qualification_complete"] is False
    assert result["g1_complete"] is False


def test_formal_analyzer_rejects_a_failed_balanced_cell(monkeypatch):
    monkeypatch.setattr(
        formal_analyzer,
        "validate_paired_block_evidence",
        lambda *args, **kwargs: {
            "classification": "PASS_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_EVIDENCE",
            "proposed_contract_passed": True,
            "profile_count": 48,
            "execution_sequence_validated": True,
        },
    )
    monkeypatch.setattr(
        formal_analyzer,
        "_balanced_groups",
        lambda *args, **kwargs: [
            {"family": f"family-{index}", "grid": "grid", "passed": index != 3}
            for index in range(4)
        ],
    )
    result = formal_analyzer.analyze(Path("profiles"), Path("progress"))
    assert result["classification"] == formal_analyzer.FAIL_CLASSIFICATION
    assert result["proposed_contract_passed"] is False


def _runner_inputs(tmp_path: Path) -> dict[str, Path]:
    baseline = tmp_path / "baseline" / "bin" / "python"
    candidate = tmp_path / "candidate" / "bin" / "python"
    baseline.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    baseline.write_bytes(b"python\n")
    candidate.write_bytes(b"python\n")
    support = tmp_path / "support"
    for relative in SUPPORT_FILES:
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    return {"baseline": baseline, "candidate": candidate, "support": support}


def _authorization(support: Path, **overrides) -> dict[str, object]:
    result = {
        "schema": AUTHORIZATION_SCHEMA,
        "formal_contract_adoption_authorized": True,
        "single_h100_submission_authorized": True,
        "automatic_retry_authorized": False,
        "maximum_formal_h100_submissions": 1,
        "source_sha256": {
            relative: _sha256(support / relative) for relative in BOUND_SOURCE_FILES
        },
    }
    result.update(overrides)
    return result


def test_runner_plans_48_profiles_but_execution_requires_bound_authority(tmp_path):
    inputs = _runner_inputs(tmp_path)
    arguments = {
        "baseline_python": inputs["baseline"],
        "candidate_python": inputs["candidate"],
        "support_root": inputs["support"],
        "output_root": tmp_path / "output",
    }
    command_plan = build_command_plan(**arguments)
    preflight(command_plan, for_execution=False)
    assert command_plan["profile_count"] == 48
    assert command_plan["progress_schema"] == PROGRESS_SCHEMA
    assert command_plan["execution_authorization"] is None
    assert command_plan["bound_source_files"] == list(BOUND_SOURCE_FILES)
    assert not Path(command_plan["output_root"]).exists()
    with pytest.raises(PermissionError, match="requires an authorization record"):
        preflight(command_plan, for_execution=True)

    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            _authorization(inputs["support"], single_h100_submission_authorized=False)
        ),
        encoding="utf-8",
    )
    unauthorized = build_command_plan(**arguments, authorization_record=authorization)
    with pytest.raises(PermissionError, match="single H100 submission"):
        preflight(unauthorized, for_execution=True)

    authorization.unlink()
    authorization.write_text(
        json.dumps(_authorization(inputs["support"])), encoding="utf-8"
    )
    authorized = build_command_plan(**arguments, authorization_record=authorization)
    preflight(authorized, for_execution=True)
    assert authorized["execution_authorization"]["sha256"] == _sha256(authorization)

    target = inputs["support"] / BOUND_SOURCE_FILES[-1]
    target.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity differs"):
        preflight(authorized, for_execution=True)


def test_h100_wrapper_is_prepared_but_cannot_mark_global_complete():
    job = (ROOT / "jobs/rc5_g1_balanced_crossover_h100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:H100:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "run_rc5_g1_checkpoint_tmp_preflight.py" in job
    assert "run_rc5_g1_environment.py" in job
    assert "run_rc5_g1_balanced_crossover_profile_matrix.py" in job
    assert "--authorization-record" in job
    assert "test ! -e \"${RC5_G1_OUTPUT_ROOT}\"" in job
    assert "BALANCED_CROSSOVER_COMPLETE" in job
    assert "touch \"${RC5_G1_OUTPUT_ROOT}/COMPLETE\"" not in job


def test_adoption_record_binds_sources_and_requires_new_execution_authority():
    record = json.loads(
        (
            ROOT
            / "notes"
            / "PSSolver_v0_2_0rc5_g1_balanced_crossover_adoption_v15.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == CLASSIFICATION
    assert record["parent_commit"] == (
        "bd129703e6bbc585d0b47a6dddc320a7894551ff"
    )
    assert record["adoption"]["formal_v13_classification_preserved"] == (
        "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE"
    )
    assert record["adoption"]["formal_v13_pass_granted"] is False
    support = record["qualification_support"]
    assert support["source_bound_authorization_required"] is True
    assert support["authorization_must_bind_exactly_five_source_files"] is True
    assert support["exactly_one_h100_submission_must_be_separately_authorized"] is True
    cpu = record["cpu_validation"]
    assert cpu["targeted_balanced_contract_and_qualification_tests"] == {
        "passed": 22,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert cpu["all_rc5_g1_tests"] == {
        "passed": 206,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert cpu["formal_analyzer_v13_profile_smoke"]["formal_qualification_claimed"] is False
    assert cpu["git_diff_check"] == "PASS"
    assert cpu["cache_count"] == 0
    authorization = record["authorization"]
    assert authorization["formal_contract_adoption_authorized"] is True
    assert authorization["push_authorized"] is False
    assert authorization["h100_submission_authorized"] is False
    assert authorization["maximum_h100_submissions_without_new_authorization"] == 0
    assert authorization["automatic_retry_authorized"] is False
    assert record["scope"]["formal_requalification_performed"] is False
    assert record["scope"]["qualification_complete"] is False
    assert record["scope"]["g1_complete"] is False
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
