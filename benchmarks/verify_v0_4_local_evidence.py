#!/usr/bin/env python3
"""Fail-closed verifier for local evidence removed from the pytest suite.

The public test suite validates the repository-side, content-addressed contract.
This tool separately verifies the historical local files named by that contract.
It intentionally uses only the Python standard library and never writes below
an evidence root.  The only output it may create is the explicitly requested
machine-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "notes/PSSolver_v0_4_0_s16_r8_local_evidence_catalog.json"
SCHEMA = "pssolver.v0_4.s16.r8.local_evidence_verification.v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _resolve_historical_path(raw: str, home_root: Path) -> Path:
    original = PurePosixPath(raw)
    home = PurePosixPath("/home/fansenwei")
    try:
        relative = original.relative_to(home)
    except ValueError as exc:
        raise ValueError(f"catalog path is outside /home/fansenwei: {raw}") from exc
    return home_root.joinpath(*relative.parts)


def _manifest_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"invalid checksum row {path}:{line_number}")
        digest = line[:64]
        relative = line[66:]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid digest at {path}:{line_number}")
        pure = PurePosixPath(relative)
        if ".." in pure.parts:
            raise ValueError(f"unsafe checksum path at {path}:{line_number}")
        rows.append((digest, relative))
    return rows


def _check_manifest(
    path: Path, expected_entries: int, home_root: Path
) -> dict[str, Any]:
    rows = _manifest_rows(path)
    if len(rows) != expected_entries:
        raise ValueError(
            f"manifest entry count differs for {path}: "
            f"{len(rows)} != {expected_entries}"
        )
    failures: list[dict[str, str]] = []
    for expected, relative in rows:
        pure = PurePosixPath(relative)
        if pure.is_absolute():
            try:
                below_home = pure.relative_to(PurePosixPath("/home/fansenwei"))
            except ValueError as exc:
                raise ValueError(
                    f"absolute checksum path is outside /home/fansenwei: {relative}"
                ) from exc
            target = home_root.joinpath(*below_home.parts)
        else:
            target = path.parent / relative
        if not target.is_file():
            failures.append({"path": str(target), "error": "missing_regular_file"})
            continue
        observed = _sha256(target)
        if observed != expected:
            failures.append(
                {
                    "path": str(target),
                    "error": "sha256_mismatch",
                    "expected": expected,
                    "observed": observed,
                }
            )
    if failures:
        raise ValueError(
            f"manifest content verification failed for {path}: "
            f"{json.dumps(failures[:5], sort_keys=True)}"
        )
    return {"entries": len(rows), "all_entries_verified": True}


def _check_tar_members(path: Path, forbidden: list[str]) -> dict[str, Any]:
    with tarfile.open(path, "r:*") as archive:
        names = archive.getnames()
    violations = [
        name
        for name in names
        if any(fragment in f"/{name.lstrip('/')}" for fragment in forbidden)
    ]
    if violations:
        raise ValueError(f"forbidden tar members in {path}: {violations[:5]}")
    return {"members": len(names), "forbidden_member_matches": 0}


def _check_stop_analysis(home_root: Path) -> dict[str, Any]:
    from benchmarks.analyze_v0_3_s09_cpu_6x60_stop import (
        EXPECTED_CANDIDATE_SHA256,
        EXPECTED_CAUSAL_SHA256,
        EXPECTED_SELF_SHA256,
        _read_bound,
        analyze_existing_evidence,
    )

    evidence = home_root / "pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3"
    prior = home_root / "pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2"
    result = analyze_existing_evidence(
        _read_bound(evidence / "results/self_control/matrix.json", EXPECTED_SELF_SHA256),
        _read_bound(
            evidence / "results/candidate_bound/matrix.json",
            EXPECTED_CANDIDATE_SHA256,
        ),
        _read_bound(
            prior / "results/s09_diagnosis_c105_cost_5x30/matrix.json",
            EXPECTED_CAUSAL_SHA256,
        ),
    )
    expected = {
        "classification": "TRANSIENT_PROFILE_SLOWDOWN_SUPPORTED_CAUSE_UNRESOLVED",
        "trial": 1,
        "qualification_failure_remains_binding": True,
        "same_pressure_iteration_count": True,
        "remaining_trials_all_below_1_05": True,
        "unique_runtime_root_cause_supported": False,
    }
    observed = {
        "classification": result["classification"],
        "trial": result["failed_profile"]["trial"],
        "qualification_failure_remains_binding": result["adjudication"][
            "qualification_failure_remains_binding"
        ],
        "same_pressure_iteration_count": result["failed_profile"][
            "same_pressure_iteration_count"
        ],
        "remaining_trials_all_below_1_05": result["failed_profile"][
            "remaining_trials_all_below_1_05"
        ],
        "unique_runtime_root_cause_supported": result["adjudication"][
            "unique_runtime_root_cause_supported"
        ],
    }
    if observed != expected:
        raise ValueError(f"S-09 stop analysis differs: {observed}")
    return observed


def assert_pytest_migration(nodeid: str) -> dict[str, Any]:
    """Assert that one historical pytest check is represented in the catalog."""

    catalog = _load(CATALOG)
    entry = catalog["pytest_migrations"].get(nodeid)
    if not isinstance(entry, dict):
        raise AssertionError(f"missing R8 migration entry for {nodeid}")
    check_ids = entry.get("check_ids")
    if not isinstance(check_ids, list) or not check_ids:
        raise AssertionError(f"empty R8 migration entry for {nodeid}")
    known = catalog["checks"]
    missing = [check_id for check_id in check_ids if check_id not in known]
    if missing:
        raise AssertionError(f"unknown R8 checks for {nodeid}: {missing}")
    if entry.get("pytest_reads_external_paths") is not False:
        raise AssertionError(f"R8 migration does not forbid pytest I/O for {nodeid}")
    return entry


def verify(catalog_path: Path, home_root: Path) -> dict[str, Any]:
    catalog = _load(catalog_path)
    results: list[dict[str, Any]] = []
    passed = True
    for check_id, check in catalog["checks"].items():
        path = _resolve_historical_path(check["path"], home_root)
        result: dict[str, Any] = {
            "id": check_id,
            "kind": check["kind"],
            "historical_path": check["path"],
            "resolved_path": str(path),
            "passed": False,
        }
        try:
            if not path.is_file():
                raise FileNotFoundError(f"missing regular file: {path}")
            observed_size = path.stat().st_size
            observed_sha256 = _sha256(path)
            if observed_size != check["size"]:
                raise ValueError(
                    f"size mismatch for {path}: {observed_size} != {check['size']}"
                )
            if observed_sha256 != check["sha256"]:
                raise ValueError(
                    f"sha256 mismatch for {path}: "
                    f"{observed_sha256} != {check['sha256']}"
                )
            result.update(size=observed_size, sha256=observed_sha256)
            if check["kind"] == "manifest":
                result["manifest"] = _check_manifest(
                    path, check["entries"], home_root
                )
            if "forbidden_tar_fragments" in check:
                result["tar"] = _check_tar_members(
                    path, check["forbidden_tar_fragments"]
                )
            result["passed"] = True
        except Exception as exc:  # report every missing or inconsistent item
            passed = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)

    semantic: list[dict[str, Any]] = []
    try:
        semantic.append(
            {
                "id": "S09_STOP_ANALYSIS",
                "passed": True,
                "observed": _check_stop_analysis(home_root),
            }
        )
    except Exception as exc:
        passed = False
        semantic.append(
            {
                "id": "S09_STOP_ANALYSIS",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    covered = {
        check_id
        for entry in catalog["pytest_migrations"].values()
        for check_id in entry["check_ids"]
    }
    if covered != set(catalog["checks"]):
        passed = False
        semantic.append(
            {
                "id": "CROSSWALK_TOTALITY",
                "passed": False,
                "unmapped_checks": sorted(set(catalog["checks"]) - covered),
                "unknown_checks": sorted(covered - set(catalog["checks"])),
            }
        )
    else:
        semantic.append(
            {
                "id": "CROSSWALK_TOTALITY",
                "passed": True,
                "migrated_tests": len(catalog["pytest_migrations"]),
                "checks": len(catalog["checks"]),
            }
        )
    return {
        "schema": SCHEMA,
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": _sha256(catalog_path),
        "home_root": str(home_root.resolve()),
        "check_count": len(results),
        "passed_check_count": sum(bool(row["passed"]) for row in results),
        "failed_check_count": sum(not bool(row["passed"]) for row in results),
        "checks": results,
        "semantic_checks": semantic,
        "passed": passed,
    }


def _atomic_write(path: Path, payload: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--home-root", type=Path, default=Path("/home/fansenwei"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.catalog.resolve(), args.home_root.resolve())
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _atomic_write(args.output.resolve(), payload)
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
