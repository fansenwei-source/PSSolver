"""Bindings for the v0.3 N-1 / V-17 checkpoint integration candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor

import pssolver


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
S01_RECORD = ROOT / "notes/PSSolver_v0_3_0_s01_registry_migration.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_integration_keeps_the_v03_version_and_exact_runtime_scope() -> None:
    record = _read()
    assert record["version"] == "0.3.0.dev0"
    assert pssolver.__version__ == "0.3.0"
    assert record["classification"] == (
        "PASS_LOCAL_V0_3_N1_V17_CHECKPOINT_INTEGRATION_CANDIDATE"
    )
    assert record["scope"]["runtime_delta_paths"] == [
        "pssolver/io/checkpoint.py"
    ]
    assert record["scope"]["checkpoint_test_paths"] == [
        "tests/test_v0_2_1_n1_checkpoint_nfs.py",
        "tests/test_v0_2_2_v17_checkpoint_symlink_parent.py",
    ]
    relative = "pssolver/io/checkpoint.py"
    expected = record["runtime_sources"][relative]
    assert _sha256(ROOT / relative) == resolve_s03_successor(
        relative, expected
    )


def test_integration_binds_the_original_and_development_commits() -> None:
    source = _read()["source"]
    assert source["n1_original_commit"] == (
        "927dc89aa63bcee9b03248f52662389bc947c47a"
    )
    assert source["v17_original_commit"] == (
        "611d167d1b2c2d634222f608f4c9cd4dc671636d"
    )
    for commit in (
        source["development_parent"],
        source["v0_2_2_release_source"],
        source["v0_2_2_evidence_tip"],
        source["n1_development_cherry_pick"],
        source["v17_development_cherry_pick"],
    ):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=True,
        )


def test_integration_binds_runtime_tests_and_forward_successors() -> None:
    record = _read()
    s01_successors = json.loads(S01_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    for relative, expected in record["regression_tests"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, expected
        )
    release_tip = record["source"]["v0_2_2_evidence_tip"]
    for relative, successor in record["historical_successor_sources"].items():
        historical = _git_bytes(release_tip, relative)
        assert hashlib.sha256(historical).hexdigest() == successor[
            "historical_sha256"
        ]
        expected = successor["current_sha256"]
        s01_successor = s01_successors.get(relative)
        if (
            s01_successor is not None
            and s01_successor["historical_sha256"] == expected
        ):
            expected = s01_successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, expected
        )
        assert successor["reason"]


def test_integration_preserves_public_numerical_and_release_boundaries() -> None:
    record = _read()
    scope = record["scope"]
    for key in (
        "checkpoint_schema_changed",
        "provider_identity_changed",
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "production_default_changed",
    ):
        assert scope[key] is False
    strategy = record["integration_strategy"]
    assert strategy["release_history_rewritten"] is False
    assert strategy["development_history_rewritten"] is False
    assert strategy["merge_commit_used"] is True
    authorization = record["authorization"]
    assert authorization["local_integration_authorized"] is True
    assert authorization["external_review_requested"] is True
    assert authorization["push_authorized"] is False
    assert authorization["gpu_authorized"] is False
    assert authorization["h100_authorized"] is False


def test_integration_records_the_next_v03_planning_boundary() -> None:
    plan = _read()["development_plan"]
    assert plan["remaining_v0_2_x_low_and_info_items"] == (
        "handle_on_0_3_line_without_a_separate_v0_2_3"
    )
    assert plan["NEW-14"] == (
        "schedule_gpu_backward_tf32_scope_verification_for_0_3_with_separate_"
        "gpu_or_h100_authorization"
    )
    assert plan["gpu_authorized"] is False
    assert plan["h100_authorized"] is False
