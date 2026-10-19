"""Record checks for the RC6 H100 stage-producer freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_h100_stage_producers.json"
SUCCESSOR_RECORD = (
    ROOT / "notes/PSSolver_v0_2_0rc6_pressure_smoke_contract_correction.json"
)
CHECKPOINT_SUCCESSOR_RECORD = (
    ROOT / "notes/PSSolver_v0_2_0rc6_checkpoint_work_root_correction.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_producer_record_binds_every_qualification_source():
    record = _record()
    successors: dict[str, object] = {}
    for path in (SUCCESSOR_RECORD, CHECKPOINT_SUCCESSOR_RECORD):
        registered = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ]
        for relative, registration in registered.items():
            previous = successors.get(relative)
            if previous is not None:
                assert registration["predecessor_sha256"] == previous[
                    "successor_sha256"
                ]
            successors[relative] = registration
    assert record["status"] == "local_stage_producers_complete_h100_not_authorized"
    for relative, expected in record["source_sha256"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            registration = successors.get(relative)
            assert registration is not None
            assert registration["predecessor_sha256"] == expected
            assert registration["successor_sha256"] == actual


def test_stage_producer_record_freezes_exact_order_and_fail_closed_execution():
    contract = _record()["stage_contract"]
    assert contract["ordered_stages"] == [
        "environment", "cuda_only", "contract_smokes", "tf32_scope",
        "current_checkpoints", "released_checkpoints", "numerical", "profiles",
        "consumer", "assemble", "analyze",
    ]
    assert contract["stage_count"] == 11
    for key in (
        "shell_free_argv", "python_executable_and_helper_sha256_frozen_at_plan_creation",
        "fail_fast", "atomic_json",
    ):
        assert contract[key] is True
    assert contract["temporary_json_residue_allowed"] is False
    assert contract["automatic_retry"] is False


def test_stage_producer_record_does_not_claim_h100_or_release_authority():
    record = _record()
    assert record["local_validation"]["h100_executed"] is False
    assert record["local_validation"]["focused_tests"] == {
        "passed": 62, "failed": 0, "skipped": 0, "xfailed": 0,
    }
    assert record["local_validation"]["full_cpu_suite"] == {
        "passed": 3994,
        "failed": 0,
        "skipped": 1,
        "deselected": 7,
        "xfailed": 0,
        "subtests_passed": 8,
        "seconds": 285.91,
    }
    assert all(record["remaining_before_submission"].values())
    authorization = record["authorization"]
    assert authorization == {
        "push_executed": False,
        "h100_submission_count": 0,
        "h100_authorized": False,
        "tag_created": False,
        "release_created": False,
        "pypi_published": False,
    }


def test_stage_producer_scope_is_qualification_only():
    scope = _record()["scope"]
    assert scope["qualification_only"] is True
    assert all(value is False for key, value in scope.items() if key != "qualification_only")
