"""D-0016 sharing and semantic no-growth ratchet for benchmark harnesses."""

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
INVENTORY_V2 = (
    ROOT / "notes/PSSolver_v0_4_0_s16_r8_benchmark_harness_sharing_v2.json"
)
SHARED = ROOT / "benchmarks/_artifact_io.py"
MIGRATED = {
    "benchmarks/verify_v0_4_local_evidence.py",
    "benchmarks/verify_v0_4_s16_numerical_equivalence.py",
}
HELPER_NAMES = {"_write_json", "write_json", "write_report", "write_artifacts"}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_helpers(source: str) -> set[str]:
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


def _semantic_helpers(source: str) -> dict[str, list[str]]:
    """Classify local artifact/provenance helpers by behavior, not only names."""

    tree = ast.parse(source)
    hits: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        segment = ast.get_source_segment(source, node) or ""
        kinds: list[str] = []
        if "os.replace(" in segment or (
            ".replace(" in segment and "tmp" in segment.lower()
        ):
            kinds.append("atomic_publish")
        if "rev-parse" in segment or (
            '"git"' in segment and "subprocess" in segment
        ):
            kinds.append("git_identity")
        if "sha256(" in segment and (
            "open(" in segment or "read_bytes" in segment
        ):
            kinds.append("file_sha256")
        has_environment_value = any(
            token in segment
            for token in (
                "sys.version",
                "sys.executable",
                "platform.platform",
                "platform.python_version",
            )
        )
        returns_environment = any(
            token in segment
            for token in ('"environment"', '"python"', '"platform"', "return {")
        )
        if has_environment_value and returns_environment:
            kinds.append("running_environment")
        if kinds:
            hits[node.name] = kinds
    return hits


def _git_show(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _combined_rows() -> list[dict[str, object]]:
    return [
        *_read(INVENTORY)["scripts"],
        *_read(INVENTORY_V2)["additional_scripts"],
    ]


def _unlisted_semantic_helper_paths(
    root: Path, allowed: set[str]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for path in sorted((root / "benchmarks").rglob("*.py")):
        if path.name == "_artifact_io.py":
            continue
        source = path.read_text(encoding="utf-8")
        semantic = _semantic_helpers(source)
        legacy = _legacy_helpers(source)
        if not semantic and not legacy:
            continue
        relative = str(path.relative_to(root))
        if relative not in allowed:
            result[relative] = semantic or {
                name: ["legacy_named_helper"] for name in sorted(legacy)
            }
    return result


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


def test_v2_inventory_reconstructs_recursive_semantic_baseline():
    base = _read(INVENTORY)
    overlay = _read(INVENTORY_V2)
    assert file_sha256(INVENTORY) == overlay["base_inventory"]["sha256"]
    assert len(base["scripts"]) == 118
    assert len(overlay["additional_scripts"]) == 20
    rows = _combined_rows()
    assert len(rows) == overlay["combined_inventory"]["scripts"] == 138
    assert len({row["path"] for row in rows}) == len(rows)
    assert sum(row["disposition"] == "frozen_hash_bound" for row in rows) == 90
    assert sum(
        row["disposition"] == "retained_documented_exception" for row in rows
    ) == 46
    assert {
        row["path"]
        for row in rows
        if row["disposition"] == "migrated_to_shared_module"
    } == MIGRATED
    baseline = overlay["base_inventory"]["baseline_commit"]
    for row in rows:
        source = _git_show(baseline, row["path"])
        assert hashlib.sha256(source).hexdigest() == row["baseline_sha256"]
    for row in overlay["additional_scripts"]:
        source = _git_show(baseline, row["path"]).decode("utf-8")
        assert _semantic_helpers(source) == row["semantic_helpers"]


def test_every_retained_or_frozen_helper_script_is_byte_pinned():
    for row in _combined_rows():
        if row["disposition"] == "migrated_to_shared_module":
            continue
        assert file_sha256(ROOT / row["path"]) == row["baseline_sha256"]
        assert row["reason"]
        if row["disposition"] == "frozen_hash_bound":
            assert row["binding_references"]


def test_recursive_semantic_helper_allowlist_can_only_shrink():
    rows = _combined_rows()
    retained = {
        row["path"]
        for row in rows
        if row["disposition"] != "migrated_to_shared_module"
    }
    assert _unlisted_semantic_helper_paths(ROOT, retained) == {}
    for relative in MIGRATED:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert not _legacy_helpers(source)
        assert not _semantic_helpers(source)
        assert "benchmarks._artifact_io" in source or "_artifact_io" in source


def test_new_top_level_semantic_helpers_fail_the_ratchet(tmp_path: Path):
    probe = tmp_path / "benchmarks/zz_probe_new_script.py"
    probe.parent.mkdir()
    probe.write_text(
        "import hashlib, os, subprocess, tempfile\n"
        "from pathlib import Path\n"
        "def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n"
        "def _git_head(): return subprocess.run(['git', 'rev-parse', 'HEAD'])\n"
        "def _write_json_new(path):\n"
        "    fd, tmp = tempfile.mkstemp()\n"
        "    os.close(fd)\n"
        "    os.replace(tmp, path)\n",
        encoding="utf-8",
    )
    errors = _unlisted_semantic_helper_paths(tmp_path, set())
    assert errors == {
        "benchmarks/zz_probe_new_script.py": {
            "_sha256": ["file_sha256"],
            "_git_head": ["git_identity"],
            "_write_json_new": ["atomic_publish"],
        }
    }


def test_new_nested_atomic_helper_fails_the_ratchet(tmp_path: Path):
    probe = tmp_path / "benchmarks/zz_probe_dir/probe_atomic.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("def _atomic_json(path):\n    return path\n", encoding="utf-8")
    errors = _unlisted_semantic_helper_paths(tmp_path, set())
    assert errors == {
        "benchmarks/zz_probe_dir/probe_atomic.py": {
            "_atomic_json": ["legacy_named_helper"]
        }
    }


def test_v2_records_mode_change_and_confirmed_temp_pruning():
    record = _read(INVENTORY_V2)
    assert "0664 to 0600" in record["observations"][
        "numerical_verifier_mode_change"
    ]
    pruned = record["observations"]["temporary_equivalence_directories_pruned"]
    assert pruned == [
        {
            "path": "/home/fansenwei/pssolver-v04-s16-equivalence-8yMlKW",
            "size_bytes": 52_486_104,
            "file_count": 4_526,
            "tree_manifest_sha256": (
                "dd25d19da130e5df0d6b01ce2194d71150d888ec809437ca2d3cb7f458072c4a"
            ),
        },
        {
            "path": "/home/fansenwei/pssolver-v04-s16-equivalence-final-IPPkKy",
            "size_bytes": 52_399_085,
            "file_count": 4_471,
            "tree_manifest_sha256": (
                "5f13a7832561eb3d7bb9d328ae1b13b325eefac8da57bc7a693026f0722d3390"
            ),
        },
        {
            "path": "/home/fansenwei/pssolver-v04-s16-equivalence-final-TgBITZ",
            "size_bytes": 52_408_254,
            "file_count": 4_474,
            "tree_manifest_sha256": (
                "91d48c0e61c7055f5ccc2850c61e4d5acc4ae1f715ac95cee856d33d402bd298"
            ),
        },
    ]


def test_d0016_scope_does_not_touch_runtime_or_relax_gates():
    inventory = _read(INVENTORY)
    overlay = _read(INVENTORY_V2)
    assert inventory["decision"] == {
        "commit": "98d29d611ad048b3f5004e97e7ee0b8a4a76aa6c",
        "id": "D-0016",
        "selection": "option b: share only live harness support",
        "sha256": "fe2ddb0ee5ba87bc159888f95e59ea45e7ded1b293f122ee5f6d44c651e45b24",
    }
    assert all(value is False for value in overlay["scope"].values())
