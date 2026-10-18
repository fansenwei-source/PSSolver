#!/usr/bin/env python3
"""Run the immutable rc1--rc5 checkpoint fixtures under the RC6 wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc6_single_h100_qualification import PROVIDER_CANDIDATE
from benchmarks.run_rc426_cuda_tests import Results


SCHEMA = "pssolver.rc6.released_checkpoint_h100.v1"
RECORD_SHA256 = "a83a8286212d450223278fb8492c69f0a41c0c546aa303da7f55d850d14ce572"
TESTS = (
    "tests/test_rc5_legacy_functional_checkpoints.py",
    "tests/test_rc5_r53_legacy_channel_checkpoint.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("released-checkpoint JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def summarize(outcomes: dict[str, object], record: dict[str, object], unchanged: bool) -> dict[str, object]:
    lists: dict[str, list[object]] = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "collection_errors"):
        value = outcomes.get(name)
        if not isinstance(value, list):
            raise TypeError(f"pytest outcome missing: {name}")
        lists[name] = value
    if len(lists["passed"]) != 20 or any(lists[name] for name in lists if name != "passed"):
        raise ValueError("released checkpoint pytest result differs")
    matrix = record.get("released_checkpoint_matrix")
    if not isinstance(matrix, dict) or matrix.get("complete") is not True or matrix.get("primary_case_count") != 13:
        raise ValueError("released checkpoint record differs")
    if unchanged is not True:
        raise ValueError("released fixture tree changed")
    return {
        "schema": SCHEMA,
        "provider_commit": PROVIDER_CANDIDATE,
        "passed": True,
        "primary_case_count": 13,
        "pytest_passed_count": 20,
        "dispositions_match_cpu_record": True,
        "source_checkpoints_unchanged": True,
        "historical_record_sha256": RECORD_SHA256,
        "test_node_ids": sorted(str(value) for value in lists["passed"]),
    }


def execute(test_root: Path, released_record: Path, output: Path) -> dict[str, object]:
    import importlib.metadata
    import pytest

    root = test_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink() or (root / "pssolver").exists():
        raise RuntimeError("ordinary non-shadowing support root required")
    if importlib.metadata.version("pssolver") != "0.2.0rc6":
        raise RuntimeError("candidate package version differs")
    record_path = released_record.expanduser().resolve()
    if not record_path.is_file() or record_path.is_symlink() or _sha256(record_path) != RECORD_SHA256:
        raise RuntimeError("released checkpoint record identity differs")
    fixture_roots = (
        root / "tests/fixtures/functional_checkpoint_releases_v1",
        root / "tests/fixtures/legacy_channel_checkpoints_v1",
    )
    before = {str(path): _tree(path) for path in fixture_roots}
    collector = Results()
    previous = Path.cwd()
    try:
        os.chdir(root)
        exit_code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *TESTS], plugins=[collector]))
    finally:
        os.chdir(previous)
    if exit_code != 0:
        raise RuntimeError(f"released checkpoint pytest failed: {exit_code}")
    after = {str(path): _tree(path) for path in fixture_roots}
    result = summarize(collector.metadata(), json.loads(record_path.read_text(encoding="utf-8")), before == after)
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--released-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.test_root, args.released_record, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
