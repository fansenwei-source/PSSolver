#!/usr/bin/env python3
"""Adjudicate the RC6 PSSolver-Control installed-wheel CPU review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


PROVIDER_WHEEL_SHA256 = (
    "8840ebfaa44a20a828928256c319edfa2aa4a9b8adf4f1bac014c33964ba0dd6"
)
CONSUMER_ARCHIVE_SHA256 = (
    "e8b3a3b14c5e51d485d2081bd8b8bbc71adb057884e553ac4726346a35adfe39"
)
CONSUMER_WHEEL_SHA256 = (
    "eca0c2c1e9b39de1f653eb1d0c1dc69a6f49c72736da758eea814205f115be0f"
)
ADAPTER_SHA256 = (
    "46b1b3a9028d2c16282c69f3ec246fc1571fe9fdf2285ba26107eacc5a80f6ef"
)
EXPECTED_SKIPS = {
    ("tests.test_checkpoint_memory", "test_reverse_sweep_holds_one_segment[20]"),
    ("tests.test_checkpoint_memory", "test_reverse_sweep_holds_one_segment[100]"),
    (
        "tests.test_gpu_optional",
        "test_cuda_adjoint_matches_autograd_with_cpu_checkpoints",
    ),
    ("tests.test_gpu_optional", "test_cuda_and_cpu_toy_agree_to_rounding"),
    (
        "tests.test_gpu_optional",
        "test_cpu_parameters_on_cuda_problem_fail_explicitly",
    ),
    ("tests.test_warmup_experiments", "test_r5_benchmark"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _junit(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    _require(bool(suites), f"no test suite in {path}")
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    skips = {
        (case.attrib.get("classname", ""), case.attrib.get("name", ""))
        for case in root.iter("testcase")
        if case.find("skipped") is not None
    }
    return {"counts": counts, "skips": sorted([list(value) for value in skips])}


def _audit(path: Path, purelib: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value["passed"] is True, f"import audit failed: {path}")
    _require(value["modules_outside_site_packages"] == {}, "module escaped purelib")
    _require(value["shadowing_sys_path_entries"] == [], "source shadow path present")
    _require(value["worktree_sys_path_entries"] == [], "worktree path present")
    for name in ("pssolver_file", "pssolver_control_file"):
        _require(
            Path(value[name]).resolve().is_relative_to(purelib),
            f"{name} is outside qualification purelib",
        )
    return value


def analyze(arguments: argparse.Namespace) -> None:
    source_targeted = _junit(arguments.source_targeted)
    source_full = _junit(arguments.source_full)
    installed_targeted = _junit(arguments.installed_targeted)
    installed_full = _junit(arguments.installed_full)
    expected_targeted = {
        "tests": 87,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    expected_full = {
        "tests": 345,
        "failures": 0,
        "errors": 0,
        "skipped": 6,
    }
    _require(source_targeted["counts"] == expected_targeted, "source targeted differs")
    _require(installed_targeted["counts"] == expected_targeted, "installed targeted differs")
    _require(source_full["counts"] == expected_full, "source full suite differs")
    _require(installed_full["counts"] == expected_full, "installed full suite differs")
    expected_skips = sorted([list(value) for value in EXPECTED_SKIPS])
    _require(source_full["skips"] == expected_skips, "source skip identities differ")
    _require(installed_full["skips"] == expected_skips, "installed skip identities differ")

    purelib = arguments.purelib.expanduser().resolve()
    targeted_audit = _audit(arguments.targeted_audit, purelib)
    full_audit = _audit(arguments.full_audit, purelib)
    _require(_sha256(arguments.provider_wheel) == PROVIDER_WHEEL_SHA256, "Provider wheel differs")
    _require(
        _sha256(arguments.consumer_archive) == CONSUMER_ARCHIVE_SHA256,
        "Consumer archive differs",
    )
    _require(
        _sha256(arguments.consumer_wheel) == CONSUMER_WHEEL_SHA256,
        "Consumer wheel differs",
    )
    _require(_sha256(arguments.source_adapter) == ADAPTER_SHA256, "source adapter differs")
    _require(_sha256(arguments.installed_adapter) == ADAPTER_SHA256, "installed adapter differs")
    provisioning = json.loads(arguments.provisioning.read_text(encoding="utf-8"))
    _require(provisioning["selected_environment"]["pip_check"] == "PASS", "pip check failed")
    _require(
        provisioning["selected_environment"]["target_packages_in_own_purelib"] is True,
        "target wheels are not installed in the selected purelib",
    )
    _require(
        all(value["qualification_evidence"] is False for value in provisioning["rejected_attempts"]),
        "a rejected setup attempt was treated as evidence",
    )

    result = {
        "schema": "pssolver.rc6.control_installed_cpu_analysis.v1",
        "classification": "PASS_RC6_PSSOLVER_CONTROL_GUARD_ORDER_INSTALLED_WHEEL_CPU",
        "provider_commit": "ea801ec08f5b2465661334e1c126133e55b68e96",
        "consumer_commit": "208edd03e6b26a0e788c07d295fddd7a67237189",
        "consumer_runtime_implementation_commit": (
            "6d3558b1367da9a3e7d6b3f8fbc5584cf384bfe3"
        ),
        "source_gate": {
            "targeted": source_targeted,
            "full_cpu": source_full,
            "passed": True,
        },
        "installed_wheel_gate": {
            "targeted": installed_targeted,
            "full_cpu": installed_full,
            "targeted_import_audit": targeted_audit,
            "full_import_audit": full_audit,
            "pip_check": "PASS",
            "source_shadow_import": False,
            "passed": True,
        },
        "artifacts": {
            "provider_wheel_sha256": PROVIDER_WHEEL_SHA256,
            "consumer_archive_sha256": CONSUMER_ARCHIVE_SHA256,
            "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
            "source_and_installed_adapter_sha256": ADAPTER_SHA256,
        },
        "guard_order_regression_passed": True,
        "provider_identity_migration_required": False,
        "functional_api_version": "1.0",
        "h100_required": False,
        "h100_submission_count": 0,
        "production_default_changed": False,
        "nematics3d_modified": False,
        "passed": True,
    }
    _atomic_json(arguments.output, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in (
        "source-targeted",
        "source-full",
        "installed-targeted",
        "installed-full",
        "targeted-audit",
        "full-audit",
        "provider-wheel",
        "consumer-archive",
        "consumer-wheel",
        "source-adapter",
        "installed-adapter",
        "provisioning",
        "purelib",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main() -> None:
    analyze(_parser().parse_args())


if __name__ == "__main__":
    main()
