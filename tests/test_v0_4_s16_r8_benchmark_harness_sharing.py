"""D-0016 sharing and no-growth ratchet for active benchmark harnesses."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks._artifact_io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    provenance_record,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "notes/PSSolver_v0_4_0_s16_r8_benchmark_harness_sharing.json"
SHARED = ROOT / "benchmarks/_artifact_io.py"
MIGRATED = {
    "benchmarks/verify_v0_4_local_evidence.py",
    "benchmarks/verify_v0_4_s16_numerical_equivalence.py",
}
HELPER_NAMES = {"_write_json", "write_json", "write_report", "write_artifacts"}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _helpers(source: str) -> set[str]:
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        name
        for name in names
        if "atomic" in name.lower()
        or "provenance" in name.lower()
        or name in HELPER_NAMES
    }


def _git_show(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_shared_module_is_standard_library_only_and_outside_runtime():
    source = SHARED.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names | {"__future__"}
    assert not SHARED.is_relative_to(ROOT / "pssolver")


def test_shared_atomic_writers_preserve_explicit_bytes(tmp_path: Path):
    report = tmp_path / "report.json"
    atomic_write_json(
        report,
        {"z": 1, "a": [2.0]},
        allow_nan=False,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    assert report.read_bytes() == b'{\n  "a": [\n    2.0\n  ],\n  "z": 1\n}\n'
    text = tmp_path / "report.txt"
    atomic_write_text(text, "alpha\nbeta\n", overwrite=False)
    assert text.read_bytes() == b"alpha\nbeta\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_text(text, "changed", overwrite=False)
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_shared_provenance_records_git_files_and_environment():
    record = provenance_record(ROOT, [SHARED])
    assert record["git"]["head"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert record["files"] == [
        {
            "path": str(SHARED.resolve()),
            "size": SHARED.stat().st_size,
            "sha256": file_sha256(SHARED),
        }
    ]
    assert record["environment"]["python_executable"] == str(
        Path(sys.executable).resolve()
    )


def test_inventory_reconstructs_the_conservative_baseline():
    inventory = _read(INVENTORY)
    baseline = inventory["baseline"]["commit"]
    rows = inventory["scripts"]
    assert len(rows) == 118
    assert len({row["path"] for row in rows}) == len(rows)
    assert sum(row["disposition"] == "frozen_hash_bound" for row in rows) == 87
    assert sum(
        row["disposition"] == "retained_documented_exception" for row in rows
    ) == 29
    assert {
        row["path"]
        for row in rows
        if row["disposition"] == "migrated_to_shared_module"
    } == MIGRATED
    for row in rows:
        source = _git_show(baseline, row["path"])
        assert hashlib.sha256(source).hexdigest() == row["baseline_sha256"]
        assert sorted(_helpers(source.decode("utf-8"))) == row["helper_names"]


def test_frozen_helpers_are_byte_identical_and_exceptions_are_reasoned():
    rows = _read(INVENTORY)["scripts"]
    for row in rows:
        if row["disposition"] == "frozen_hash_bound":
            assert file_sha256(ROOT / row["path"]) == row["baseline_sha256"]
            assert row["binding_references"]
        if row["disposition"] == "retained_documented_exception":
            assert row["reason"]


def test_local_helper_allowlist_can_only_shrink():
    rows = _read(INVENTORY)["scripts"]
    retained = {
        row["path"]
        for row in rows
        if row["disposition"] != "migrated_to_shared_module"
    }
    observed = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "benchmarks").glob("*.py")
        if path != SHARED and _helpers(path.read_text(encoding="utf-8"))
    }
    assert observed <= retained
    assert not (observed & MIGRATED)
    for relative in MIGRATED:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "benchmarks._artifact_io" in source or "_artifact_io" in source


def test_d0016_scope_does_not_touch_runtime_or_relax_gates():
    inventory = _read(INVENTORY)
    assert inventory["decision"] == {
        "commit": "98d29d611ad048b3f5004e97e7ee0b8a4a76aa6c",
        "id": "D-0016",
        "selection": "option b: share only live harness support",
        "sha256": "fe2ddb0ee5ba87bc159888f95e59ea45e7ded1b293f122ee5f6d44c651e45b24",
    }
    assert inventory["scope"] == {
        "gpu_specific_path_changed": False,
        "historical_script_rewritten": False,
        "numerical_output_changed": False,
        "public_interface_changed": False,
        "runtime_changed": False,
        "skip_or_xfail_added": False,
        "threshold_changed": False,
    }
