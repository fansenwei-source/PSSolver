"""Bindings for the S-10 R-0033 re-review candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT / "notes/PSSolver_v0_3_0_s03_successor_56_s10_r0033_candidate.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
V020_ORACLE = (
    ROOT / "tests/oracles/v0_2_0_checkpoint_continuations_v1/manifest.json"
)
S08_MANIFEST = ROOT / "tests/fixtures/historical_cpu_coverage_v1/manifest.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0033_candidate_binds_exact_local_implementation() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["git_identity"]["implementation_commit"] == (
        "61838af7910d955b27268abd4a79e4d32f235f58"
    )
    assert record["git_identity"]["status_record_commit"] == (
        "c37991ba37819d88525dd774fb063ae6234cdce2"
    )
    assert record["next_request"] == "R-0033"


def test_r0033_candidate_claims_v33_and_v43_without_closing_them() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ("V-33", "V-43"):
        assert record["claims"][issue_id] == "claimed_fixed"
        assert issues[issue_id]["status"] in {"claimed_fixed", "verified"}
        assert issues[issue_id]["fixed_in"] == (
            "a54074013e5f0fb9bfe278baf944f805bf86eef4"
        )
        assert "61838af7910d955b27268abd4a79e4d32f235f58" in (
            issues[issue_id]["notes"]
        )
        if issues[issue_id]["status"] == "verified":
            assert issues[issue_id]["verified_by"] == "R-0033"
        else:
            assert "verified_by" not in issues[issue_id]


def test_r0033_candidate_binds_authentic_per_oracle_builds() -> None:
    v020 = _read(V020_ORACLE)["numerical_build_fingerprint"]
    s08 = _read(S08_MANIFEST)["generation_environment"][
        "numerical_build_fingerprint"
    ]
    assert v020["torch_version"] == "2.6.0"
    assert v020["torch_cuda_version"] == "12.6"
    assert v020["torch_git_version"] == "Unknown"
    assert v020["numpy_configuration_source"] == "mode_dicts"
    assert s08["torch_version"] == "2.6.0+cu124"
    assert s08["torch_cuda_version"] == "12.4"
    assert s08["torch_git_version"] == (
        "2236df1770800ffea5697b11b0bb0d910b2e59e1"
    )
    assert s08["numpy_configuration_source"] == "mode_dicts"
    assert v020 != s08


def test_r0033_candidate_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)
    for relative, expected in record["qualification_sources"].items():
        assert _sha256(ROOT / relative) == expected


def test_r0033_candidate_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["local_only"] is True
    assert all(value is False for value in record["authorization"].values())
    assert record["claims"]["checkpoint_payloads_changed"] is False
    assert record["claims"]["continuation_payloads_changed"] is False
