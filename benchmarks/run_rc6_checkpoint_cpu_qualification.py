#!/usr/bin/env python3
"""Run the RC6 released-checkpoint and current tamper CPU qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig
import xml.etree.ElementTree as ET


SOURCE_ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_COMMIT = "ea801ec08f5b2465661334e1c126133e55b68e96"
PACKAGE_VERSION = "0.2.0rc6"
RELEASED_RECORD_SHA256 = (
    "a83a8286212d450223278fb8492c69f0a41c0c546aa303da7f55d850d14ce572"
)
SUPPORT_SHA256 = {
    "benchmarks/run_rc5_g1_checkpoints.py": (
        "7e2d8bdbd30da0dc29725cd8be897ded79b3973e804c96027ef0fca36a8fab85"
    ),
    "benchmarks/run_rc426_cross_device_checkpoint.py": (
        "484e5e4192b79ef707d9b21307e14ef989695a8962f002ae95f95bc70d826e9c"
    ),
    "tests/test_rc5_legacy_functional_checkpoints.py": (
        "a421bfde2b5a8bb162697e6538b1f54961f1b1741d9b6d3fb73d34c843a20a8c"
    ),
    "tests/test_rc5_r53_legacy_channel_checkpoint.py": (
        "9240e35734ac5db9b786f962f04c4fe09fa4b1722da3818e37c8044a6dfc4b12"
    ),
}
RELEASED_TESTS = (
    "tests/test_rc5_legacy_functional_checkpoints.py",
    "tests/test_rc5_r53_legacy_channel_checkpoint.py",
)
FAMILIES = (
    "periodic_functional",
    "channel_functional_fixed_iteration",
    "channel_functional_tolerance",
    "legacy_channel",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_record(directory: Path) -> tuple[int, str]:
    values = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    payload = json.dumps(
        values, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return len(values), hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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
        _require(
            json.loads(path.read_text(encoding="utf-8")) == expected,
            "JSON round-trip differs",
        )
    finally:
        temporary.unlink(missing_ok=True)


def _installed_identity() -> dict[str, object]:
    import pssolver

    package_file = Path(pssolver.__file__).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    source_package = (SOURCE_ROOT / "pssolver").resolve()
    _require(package_file.is_relative_to(purelib), "pssolver is outside purelib")
    _require(
        not package_file.is_relative_to(source_package),
        "source shadow import detected",
    )
    _require(
        importlib.metadata.version("pssolver") == PACKAGE_VERSION,
        "installed distribution version differs",
    )
    _require(pssolver.__version__ == PACKAGE_VERSION, "public version differs")
    return {
        "qualified_commit": QUALIFIED_COMMIT,
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "purelib": str(purelib),
        "pssolver_version": pssolver.__version__,
        "pssolver_import": str(package_file),
        "source_package": str(source_package),
        "source_shadow_import": False,
        "python_isolated_mode": bool(sys.flags.isolated),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _add_support_root() -> None:
    for relative, expected in SUPPORT_SHA256.items():
        _require(
            _sha256(SOURCE_ROOT / relative) == expected,
            f"qualification support differs: {relative}",
        )
    sys.path.append(str(SOURCE_ROOT))
    os.chdir(SOURCE_ROOT)


def run_released_tests(junitxml: Path, identity_output: Path) -> int:
    identity = _installed_identity()
    _atomic_json(identity_output, identity)
    _add_support_root()
    import pytest

    return int(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                f"--junitxml={junitxml.expanduser().resolve()}",
                *RELEASED_TESTS,
            ]
        )
    )


def run_current(output_root: Path, output: Path, identity_output: Path) -> None:
    identity = _installed_identity()
    _atomic_json(identity_output, identity)
    _add_support_root()
    from benchmarks import run_rc5_g1_checkpoints as rc5

    rc5._tf32_off()
    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    cells: list[dict[str, object]] = []
    for family in FAMILIES:
        cell_root = root / "cells" / f"{family}_cpu_to_cpu"
        cell_root.mkdir(parents=True)
        report = (
            rc5._legacy_cell(cell_root, "cpu", "cpu")
            if family == "legacy_channel"
            else rc5._functional_cell(cell_root, family, "cpu", "cpu")
        )
        rc5._atomic_json(
            root / "reports" / f"{family}_cpu_to_cpu.json", report
        )
        _require(report.get("passed") is True, f"current cell failed: {family}")
        cells.append(report)
    result = {
        "schema": "pssolver.rc6.checkpoint_cpu_current.v1",
        "qualified_commit": QUALIFIED_COMMIT,
        "package_version": PACKAGE_VERSION,
        "family_count": len(cells),
        "direction": "cpu_to_cpu",
        "cells": cells,
        "passed": len(cells) == len(FAMILIES) and all(
            cell.get("passed") is True for cell in cells
        ),
    }
    _atomic_json(output, result)


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    _require(bool(suites), "pytest XML contains no testsuite")
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _crosscheck_fixtures(external_root: Path) -> dict[str, object]:
    repository_functional = (
        SOURCE_ROOT / "tests/fixtures/functional_checkpoint_releases_v1"
    )
    repository_legacy = (
        SOURCE_ROOT / "tests/fixtures/legacy_channel_checkpoints_v1"
    )
    cases: dict[str, tuple[Path, Path]] = {}
    for generation in ("rc1", "rc2", "rc3"):
        for family in ("periodic", "channel"):
            dirname = f"{family}_func_ck"
            cases[f"{generation}_{family}"] = (
                repository_functional / generation / dirname,
                external_root / generation / dirname,
            )
    legacy_manifest = json.loads(
        (repository_legacy / "manifest.json").read_text(encoding="utf-8")
    )
    for name, record in legacy_manifest["cases"].items():
        cases[name] = (
            repository_legacy / name,
            external_root / record["source_relative_path"],
        )
    results = {}
    for name, (repository, external) in sorted(cases.items()):
        _require(repository.is_dir(), f"repository fixture missing: {name}")
        _require(external.is_dir(), f"external fixture missing: {name}")
        repository_record = _tree_record(repository)
        external_record = _tree_record(external)
        results[name] = {
            "repository": list(repository_record),
            "external": list(external_record),
            "equal": repository_record == external_record,
        }
    _require(all(value["equal"] for value in results.values()), "fixture drift")
    return {
        "case_count": len(results),
        "passed": len(results),
        "failed": 0,
        "cases": results,
    }


def analyze(
    junitxml: Path,
    released_identity: Path,
    current_result: Path,
    current_identity: Path,
    external_root: Path,
    released_record: Path,
    installed_cpu_record: Path,
    output: Path,
) -> None:
    counts = _junit_counts(junitxml)
    _require(
        counts == {"tests": 20, "failures": 0, "errors": 0, "skipped": 0},
        f"released checkpoint pytest counts differ: {counts}",
    )
    released = json.loads(released_record.read_text(encoding="utf-8"))
    _require(
        _sha256(released_record) == RELEASED_RECORD_SHA256,
        "released RC5 record SHA-256 differs",
    )
    _require(
        released["released_checkpoint_matrix"]["primary_case_count"] == 13,
        "released checkpoint case count differs",
    )
    _require(
        released["released_checkpoint_matrix"]["complete"] is True,
        "released checkpoint matrix is incomplete",
    )
    current = json.loads(current_result.read_text(encoding="utf-8"))
    _require(current["passed"] is True, "current checkpoint matrix failed")
    _require(current["family_count"] == 4, "current family count differs")
    for cell in current["cells"]:
        for gate in ("payload_tamper", "identity_mismatch"):
            value = cell[gate]
            _require(value["rejected"] is True, f"{gate} was accepted")
            _require(value["guard_reached"] is True, f"{gate} hit wrong guard")
            _require(value["target_unchanged"] is True, f"{gate} mutated target")
        _require(
            cell["continuous_split_restart_equal"] is True,
            "restart replay differs",
        )
        _require(cell["source_checkpoint_unchanged"] is True, "source changed")
        _require(cell["fallback_used"] is False, "runtime fallback used")
    released_import = json.loads(released_identity.read_text(encoding="utf-8"))
    current_import = json.loads(current_identity.read_text(encoding="utf-8"))
    _require(released_import == current_import, "installed identities differ")
    installed_cpu = json.loads(installed_cpu_record.read_text(encoding="utf-8"))
    _require(
        installed_cpu["classification"] == "PASS_RC6_FROZEN_INSTALLED_WHEEL_CPU",
        "installed-wheel CPU prerequisite differs",
    )
    _require(
        installed_cpu["qualified_source"]["commit"] == QUALIFIED_COMMIT,
        "installed-wheel qualified source differs",
    )
    crosscheck = _crosscheck_fixtures(external_root)
    result = {
        "schema_version": 1,
        "phase": "RC6-10",
        "classification": (
            "PASS_RC6_RELEASED_CHECKPOINT_COMPATIBILITY_AND_TAMPER_CPU"
        ),
        "qualification_complete": True,
        "qualified_source": {
            "commit": QUALIFIED_COMMIT,
            "version": PACKAGE_VERSION,
            "installed_identity": released_import,
        },
        "prerequisite": {
            "installed_wheel_cpu_record": str(installed_cpu_record.resolve()),
            "installed_wheel_cpu_record_sha256": _sha256(installed_cpu_record),
            "classification": installed_cpu["classification"],
        },
        "released_checkpoint_matrix": {
            "historical_record": str(released_record.resolve()),
            "historical_record_sha256": RELEASED_RECORD_SHA256,
            "primary_case_count": 13,
            "pytest": counts,
            "external_fixture_crosscheck": crosscheck,
            "historical_dispositions_changed": False,
        },
        "current_rc6_matrix": {
            "family_count": 4,
            "direction": "cpu_to_cpu",
            "payload_tamper_rejected_before_mutation": True,
            "identity_mismatch_rejected_before_mutation": True,
            "continuous_split_restart_replay_exact": True,
            "source_checkpoints_unchanged": True,
            "fallback_used": False,
            "result": current,
        },
        "contract": {
            "checkpoint_schema_changed_in_rc6": False,
            "functional_api_version_changed_in_rc6": False,
            "production_default_changed": False,
            "provider_identity_impact_review_complete": False,
        },
        "scope": {
            "h100_submission_count": 0,
            "push_executed": False,
            "tag_created": False,
            "release_created": False,
            "pypi_published": False,
            "PSSolver_Control_modified": False,
            "nematics3d_modified": False,
        },
        "next_gate": "provider_identity_impact_review",
    }
    _atomic_json(output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    released = subparsers.add_parser("released-tests")
    released.add_argument("--junitxml", type=Path, required=True)
    released.add_argument("--identity-output", type=Path, required=True)
    current = subparsers.add_parser("current")
    current.add_argument("--output-root", type=Path, required=True)
    current.add_argument("--output", type=Path, required=True)
    current.add_argument("--identity-output", type=Path, required=True)
    adjudicate = subparsers.add_parser("analyze")
    adjudicate.add_argument("--junitxml", type=Path, required=True)
    adjudicate.add_argument("--released-identity", type=Path, required=True)
    adjudicate.add_argument("--current-result", type=Path, required=True)
    adjudicate.add_argument("--current-identity", type=Path, required=True)
    adjudicate.add_argument("--external-root", type=Path, required=True)
    adjudicate.add_argument("--released-record", type=Path, required=True)
    adjudicate.add_argument("--installed-cpu-record", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "released-tests":
        return run_released_tests(arguments.junitxml, arguments.identity_output)
    if arguments.command == "current":
        run_current(
            arguments.output_root,
            arguments.output,
            arguments.identity_output,
        )
        return 0
    analyze(
        arguments.junitxml,
        arguments.released_identity,
        arguments.current_result,
        arguments.current_identity,
        arguments.external_root,
        arguments.released_record,
        arguments.installed_cpu_record,
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
