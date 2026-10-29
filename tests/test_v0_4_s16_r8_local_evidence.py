"""Repository-side contracts for the D-0015 local evidence verifier."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from benchmarks.verify_v0_4_local_evidence import verify


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "notes/PSSolver_v0_4_0_s16_r8_local_evidence_catalog.json"
INVENTORY = ROOT / "notes/PSSolver_v0_4_0_s16_r8_inventory_v2.json"
VERIFIER = ROOT / "benchmarks/verify_v0_4_local_evidence.py"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r8_catalog_covers_the_execution_inventory_exactly() -> None:
    catalog = _read(CATALOG)
    inventory = _read(INVENTORY)
    expected = {
        f"{relative}::{name}"
        for relative, names in inventory["failing_tests"].items()
        for name in names
    }
    assert len(expected) == 35
    assert len({node.split("::", 1)[0] for node in expected}) == 27
    assert set(catalog["pytest_migrations"]) == expected
    assert all(
        entry["pytest_reads_external_paths"] is False
        and entry["check_ids"]
        for entry in catalog["pytest_migrations"].values()
    )


def test_r8_catalog_binds_the_confirmed_decision_and_fail_closed_policy() -> None:
    catalog = _read(CATALOG)
    assert catalog["decision"] == {
        "commit": "3477e9f3ee1d8f51a86300cd7153903d142fa869",
        "id": "D-0015",
        "selection": "option C plus mandatory local verifier",
        "sha256": "69eaded6bff3a9ddcea173f1924f3d272199e00893b8fa483936bd3be3af56b6",
    }
    assert catalog["policy"] == {
        "external_content_copied_into_repository": False,
        "historical_records_rewritten": False,
        "missing_or_mismatched_evidence_is_failure": True,
        "pytest_external_path_reads": False,
        "standard_library_only": True,
    }


def test_r8_every_local_check_is_mapped_and_content_addressed() -> None:
    catalog = _read(CATALOG)
    checks = catalog["checks"]
    mapped = {
        check_id
        for entry in catalog["pytest_migrations"].values()
        for check_id in entry["check_ids"]
    }
    assert set(checks) == mapped
    assert len(checks) == 103
    for check_id, check in checks.items():
        assert check_id.startswith("C")
        assert check["path"].startswith("/home/fansenwei/")
        assert check["size"] >= 0
        assert len(check["sha256"]) == 64
        assert check["kind"] in {"file", "manifest", "complete"}
        if check["kind"] == "manifest":
            assert check["entries"] > 0


def test_r8_verifier_uses_only_standard_library_and_repository_code() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "benchmarks",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "sys",
        "tarfile",
        "tempfile",
        "typing",
    }
    imported = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported <= allowed


def test_r8_verifier_fails_closed_when_local_evidence_is_hidden(tmp_path: Path) -> None:
    report = verify(CATALOG, tmp_path / "hidden-home")
    assert report["passed"] is False
    assert report["check_count"] == 103
    assert report["passed_check_count"] == 0
    assert report["failed_check_count"] == 103
    assert all(not row["passed"] for row in report["checks"])
