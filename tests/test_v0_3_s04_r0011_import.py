"""Evidence bindings for importing the R-0011 S-04 interim review."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_07_r0011_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r0011_import_binds_the_interim_result_and_complete_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "INTERIM_PASS_ELIGIBLE_FOR_REAL_NFS_R0011_IMPORTED"
    )
    assert record["git_identity"]["result_commit"] == (
        "645b4e687b4e56e6ed7c78c74595ecc28705b782"
    )
    assert record["sync_result"] == {
        "id": "R-0011",
        "path": (
            "/home/fansenwei/Desktop/Develop/PSSolver_sync/"
            "results/R-0011.result.json"
        ),
        "sha256": "0707d33de8748139b40a5c211dcf98ad4a1205c798287f514f9e8c7312a0ad5a",
        "classification": "INTERIM_PASS_ELIGIBLE_FOR_REAL_NFS",
        "release_blockers": [],
        "registry_updates": [],
        "new_findings": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 4512
    assert archive["checksums_sha256"] == (
        "d2d4b97dffda0b8a2e21b8793077c1018b8b3fcd53c7deb57b9f5911aa538cc4"
    )
    assert archive["complete_present"] is True


def test_r0011_historical_claims_advance_only_after_r0012() -> None:
    record = _read(RECORD)["review_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {"V-22", "V-16", "V-13", "V-14", "C2-17", "V-25"}
    assert set(record["preserved_claimed_fixed"]) == expected
    for issue_id in expected:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0012"
    assert issues["V-25"]["fixed_in"] == (
        "44a6fc3d88d94ccf939e78e23be3ffbac542def5"
    )
    for issue_id in expected - {"V-25"}:
        assert issues[issue_id]["fixed_in"] == (
            "468e06710e1b4881a557f0228dcb16be38791094"
        )
    assert record["eligible_for_one_separately_authorized_real_nfs_run"] is True
    assert record["s04_complete"] is False


def test_r0011_import_binds_only_the_independently_verified_hpcc_inputs() -> None:
    binding = _read(RECORD)["hpcc_binding"]
    assert binding == {
        "runtime_commit": "44a6fc3d88d94ccf939e78e23be3ffbac542def5",
        "checkpoint_py_sha256": (
            "1b690d50c85cb1ee3934908494f7fd4a6cc4df4f557f5c2663be4f374bb6bbec"
        ),
        "provider_archive_name": "provider_44a6fc3.tar",
        "provider_archive_sha256": (
            "b765e1b8d687af96223d4cf7eda75bc20eede727cadee7a11ab20428ec3d8760"
        ),
        "helper_sha256": (
            "900947f4104a7f1907d968ccad562bfe6b5bccc8c015517b1312906bbedf56f5"
        ),
        "rejected_stale_helper_sha256_prefix": "a16cc566",
        "s04_archive_root_sha256": (
            "4d938881c8950ee854ddef84c358e5481d1fc328804c19e16e268eb5c9d042f9"
        ),
        "fixture_builders_sha256": (
            "67f90f13e3ce007ee464827d91e895465a6261c65057c18854e2da5557253c76"
        ),
    }


def test_r0011_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["historical_successor_sources"] == {}
    validation = record["validation"]
    assert validation["environment_clone_source"].endswith(
        "pssolver_v0_2_1_claude_acceptance_20261020_v1/a2/venv_wheel"
    )
    assert validation["environment_clone_pruned"] is True
    assert validation["sys_path"][1] == (
        "/home/fansenwei/Desktop/Develop/PSSolver_dev_0_3"
    )
    assert validation["focused_pytest"] == "18 passed"
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["authorization"]["local_import_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
    assert record["next_request"] == "R-0012"
