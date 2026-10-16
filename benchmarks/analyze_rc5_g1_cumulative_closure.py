#!/usr/bin/env python3
"""Fail-closed CPU-only reassembly of the rc5 G-1 qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from benchmarks.analyze_rc5_g1_h100 import _validate_auxiliary
from benchmarks.plan_rc5_g1_h100 import build_plan


SCHEMA = "pssolver.rc5.g1.cumulative_closure.v1"
PASS_CLASSIFICATION = (
    "PASS_RC5_G1_CUMULATIVE_CLOSURE_WITH_"
    "BALANCED_CROSSOVER_PERFORMANCE_QUALIFICATION"
)

EXPECTED_ARCHIVES = {
    "cpu": {
        "entries": 1712,
        "manifest_sha256": (
            "6a10414a20d3c5f7c12e21e98afb594fe3008be3010388355a16e6fdb9fbe832"
        ),
    },
    "checkpoint": {
        "entries": 123,
        "manifest_sha256": (
            "6bc84435b3dab444156a87ba7ddc5981f93989c38098dfa3f9e56d8dcff9d627"
        ),
    },
    "v5": {
        "entries": 26,
        "manifest_sha256": (
            "049f7422b62618416745ca665f82ab2370f6a5f2e662ce296f332f461e3b7f71"
        ),
    },
    "v6": {
        "entries": 15,
        "manifest_sha256": (
            "74ef98002f8bd6b37f0d9e6cc001d17d9653cd6cb274afbd726273bd02eaeac2"
        ),
    },
    "v16": {
        "entries": 1814,
        "manifest_sha256": (
            "da50cbda196ace1e7a60c6974dfea40533575493096ed1971a91ef959e002de0"
        ),
    },
}

FOCUSED_CELLS = {
    ("plane_compiled_v2", "plane_R128"),
    ("channel_compiled_v2", "channel_R128x20x20"),
    ("periodic_functional", "periodic_R128"),
    ("channel_functional", "channel_R512x40x40"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("cumulative closure JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(
    root: Path,
    *,
    expected_entries: int,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    """Verify an immutable evidence archive without rewriting its manifest."""

    root = root.expanduser().resolve()
    manifest = root / "checksums.sha256"
    _require(manifest.is_file() and not manifest.is_symlink(), "checksum manifest missing")
    actual_manifest_sha256 = _sha256(manifest)
    _require(
        actual_manifest_sha256 == expected_manifest_sha256,
        "checksum manifest identity differs",
    )
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    _require(len(lines) == expected_entries, "checksum entry count differs")
    verified = 0
    for line in lines:
        digest, separator, name = line.partition("  ")
        _require(bool(separator) and len(digest) == 64 and bool(name), "invalid checksum line")
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        _require(path.is_file() and not path.is_symlink(), f"archive entry missing: {name}")
        _require(_sha256(path) == digest, f"archive checksum mismatch: {name}")
        verified += 1
    return {
        "root": str(root),
        "entries": expected_entries,
        "verified": verified,
        "failed": 0,
        "manifest_sha256": actual_manifest_sha256,
        "passed": True,
    }


def verify_v6_profile_sources(source_evidence: Mapping[str, object]) -> dict[str, object]:
    profiles = source_evidence.get("profiles")
    _require(isinstance(profiles, dict), "v6 profile source evidence missing")
    _require(profiles.get("count") == 96, "v6 profile source count differs")
    root = Path(str(profiles.get("root"))).expanduser().resolve()
    entries = profiles.get("entries")
    _require(isinstance(entries, list) and len(entries) == 96, "v6 profile entries differ")
    names: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, dict), "v6 profile entry is not an object")
        name = entry.get("name")
        digest = entry.get("sha256")
        _require(isinstance(name, str) and name not in names, "v6 profile name differs")
        _require(isinstance(digest, str) and len(digest) == 64, "v6 profile digest differs")
        path = root / name
        _require(path.is_file() and not path.is_symlink(), f"v6 source profile missing: {name}")
        _require(_sha256(path) == digest, f"v6 source profile checksum mismatch: {name}")
        names.add(name)
    return {
        "profile_count": 96,
        "verified": 96,
        "failed": 0,
        "source_profiles_modified": False,
        "passed": True,
    }


def _validate_archive_reports(reports: Mapping[str, object]) -> None:
    _require(set(reports) == set(EXPECTED_ARCHIVES), "archive verification set differs")
    for name, expected in EXPECTED_ARCHIVES.items():
        report = reports[name]
        _require(isinstance(report, dict), f"{name} archive verification missing")
        _require(report.get("passed") is True, f"{name} archive verification failed")
        _require(report.get("failed") == 0, f"{name} archive contains failures")
        _require(report.get("entries") == expected["entries"], f"{name} entry count differs")
        _require(report.get("verified") == expected["entries"], f"{name} verified count differs")
        _require(
            report.get("manifest_sha256") == expected["manifest_sha256"],
            f"{name} manifest identity differs",
        )


def _validate_cpu(record: Mapping[str, object]) -> None:
    _require(
        record.get("classification")
        == "PASS_RC5_G1_FROZEN_SOURCE_AND_INSTALLED_WHEEL_CPU",
        "CPU closure classification differs",
    )
    source = record.get("source_cpu")
    installed = record.get("installed_wheel")
    _require(isinstance(source, dict) and isinstance(installed, dict), "CPU closure missing")
    _require(source.get("passed") == 3623 and source.get("failed") == 0, "source CPU gate differs")
    applicable = installed.get("applicable_full")
    _require(isinstance(applicable, dict), "installed CPU gate missing")
    _require(applicable.get("passed") == 3609, "installed CPU pass count differs")
    _require(applicable.get("failed") == 0 and applicable.get("xfailed") == 0, "installed CPU failed")
    _require(installed.get("source_shadow_import") is False, "installed CPU source shadow import")
    _require(installed.get("pip_check") == "PASS", "installed CPU pip check failed")


def _validate_checkpoint(record: Mapping[str, object]) -> None:
    _require(
        record.get("classification")
        == "PASS_RC5_G1_RELEASED_CHECKPOINT_MATRIX_WITH_PROVIDER_IDENTITY_MIGRATION_REQUIRED",
        "checkpoint matrix classification differs",
    )
    matrix = record.get("released_checkpoint_matrix")
    impact = record.get("provider_identity_impact")
    _require(isinstance(matrix, dict) and matrix.get("complete") is True, "checkpoint matrix incomplete")
    _require(matrix.get("primary_case_count") == 13, "checkpoint primary count differs")
    _require(matrix.get("external_fixture_crosscheck") == {"passed": 10, "failed": 0}, "checkpoint crosscheck differs")
    installed = matrix.get("rc1_rc3_installed_wheel_tests")
    _require(isinstance(installed, dict), "released wheel checkpoint gate missing")
    _require(installed.get("passed") == 20 and installed.get("failed") == 0, "released wheel checkpoint gate failed")
    _require(isinstance(impact, dict) and impact.get("review_complete") is True, "identity review incomplete")


def _validate_v5(summary: Mapping[str, object], auxiliary: dict[str, object]) -> None:
    _require(
        summary.get("classification")
        == "PASS_RC5_G1_CONSUMER_H100_RECOVERY_V5_STOPPED_AT_CUMULATIVE_ANALYZER_TRANSFORM_WINDOW_CONTRACT",
        "v5 historical classification differs",
    )
    _require(summary.get("consumer_recovery_complete") is True, "v5 Consumer recovery incomplete")
    _require(summary.get("qualification_complete") is False, "v5 historical failure was rewritten")
    _require(summary.get("g1_complete") is False, "v5 historical G-1 failure was rewritten")
    _require(summary.get("scientific_failure") is False, "v5 scientific failure recorded")
    _validate_auxiliary(auxiliary, build_plan())


def _cell_set(entries: object, *, name: str) -> set[tuple[str, str]]:
    _require(isinstance(entries, list), f"{name} is not a list")
    result: set[tuple[str, str]] = set()
    for entry in entries:
        _require(isinstance(entry, dict), f"{name} entry is not an object")
        key = (str(entry.get("family")), str(entry.get("grid")))
        _require(key not in result, f"duplicate {name} cell: {key}")
        result.add(key)
    return result


def _validate_v6(
    summary: Mapping[str, object],
    performance: Mapping[str, object],
    transform: Mapping[str, object],
    profile_sources: Mapping[str, object],
) -> set[tuple[str, str]]:
    _require(
        summary.get("classification")
        == "STOPPED_RC5_G1_TRANSFORM_WINDOW_RECOVERY_V6_AT_FROZEN_PROFILE_PERFORMANCE_MEMORY_GATE",
        "v6 historical classification differs",
    )
    _require(summary.get("analysis_only") is True, "v6 was not analysis-only")
    _require(summary.get("qualification_complete") is False, "v6 historical failure was rewritten")
    _require(summary.get("g1_complete") is False, "v6 historical G-1 failure was rewritten")
    _require(summary.get("source_profiles_modified") is False, "v6 source profiles modified")
    _require(summary.get("thresholds_changed") is False, "v6 thresholds changed")
    _require(summary.get("new_h100_submission_count") == 0, "v6 submitted H100 work")
    _require(transform.get("passed") is True, "transform measurement-window gate failed")
    _require(transform.get("profile_count") == 96, "transform profile count differs")
    _require(transform.get("source_profiles_modified") is False, "transform profiles modified")
    _require(performance.get("profile_count") == 96, "v6 performance profile count differs")
    _require(performance.get("group_count") == 16, "v6 performance group count differs")
    groups = performance.get("groups")
    _require(isinstance(groups, list) and len(groups) == 16, "v6 performance groups differ")
    failed = _cell_set(performance.get("failed_groups"), name="v6 failed")
    _require(failed == FOCUSED_CELLS, "v6 failed cell set differs")
    passed = {
        (str(entry.get("family")), str(entry.get("grid")))
        for entry in groups
        if isinstance(entry, dict) and entry.get("passed") is True
    }
    _require(len(passed) == 12, "v6 passed cell count differs")
    _require(not (passed & failed), "v6 passed and failed cells overlap")
    _require(len(passed | failed) == 16, "v6 cell coverage differs")
    _require(profile_sources.get("passed") is True, "v6 profile source verification failed")
    _require(profile_sources.get("verified") == 96, "v6 source profile count differs")
    _require(profile_sources.get("source_profiles_modified") is False, "v6 source profiles changed")
    return passed


def _validate_v16(
    summary: Mapping[str, object],
    analysis: Mapping[str, object],
    validation: Mapping[str, object],
    slurm: Mapping[str, object],
) -> set[tuple[str, str]]:
    _require(
        summary.get("classification")
        == "PASS_RC5_G1_BALANCED_CROSSOVER_V16_SINGLE_H100_QUALIFICATION",
        "v16 summary classification differs",
    )
    _require(summary.get("v16_qualification_complete") is True, "v16 qualification incomplete")
    _require(summary.get("eligible_for_g1_cumulative_closure") is True, "v16 not eligible for cumulative closure")
    _require(summary.get("g1_complete") is False, "v16 improperly claimed cumulative closure")
    _require(summary.get("formal_h100_submission_count") == 1, "v16 submission count differs")
    _require(summary.get("automatic_retry") is False, "v16 automatic retry used")
    _require(summary.get("second_job_submitted") is False, "v16 second job submitted")
    _require(summary.get("profile_count") == 48, "v16 profile count differs")
    _require(
        analysis.get("classification")
        == "PASS_RC5_G1_BALANCED_CROSSOVER_PERFORMANCE_MEMORY_EVIDENCE",
        "v16 formal analyzer classification differs",
    )
    _require(analysis.get("analysis_only") is False, "v16 formal analysis marked analysis-only")
    _require(analysis.get("formal_requalification_performed") is True, "v16 requalification missing")
    _require(analysis.get("proposed_contract_passed") is True, "v16 contract failed")
    groups = analysis.get("groups")
    cells = _cell_set(groups, name="v16 focused")
    _require(cells == FOCUSED_CELLS, "v16 focused cell set differs")
    _require(all(entry.get("passed") is True for entry in groups), "v16 focused cell failed")
    checks = validation.get("checks")
    _require(isinstance(checks, dict) and checks, "v16 validation checks missing")
    _require(all(value is True for value in checks.values()), "v16 validation check failed")
    _require(validation.get("profile_count") == 48, "v16 validation profile count differs")
    _require(slurm.get("job_id") == "10886152", "v16 job identity differs")
    _require(slurm.get("state") == "COMPLETED", "v16 job did not complete")
    _require(slurm.get("exit_code") == "0:0", "v16 job exit code differs")
    _require(slurm.get("formal_h100_submission_count") == 1, "v16 Slurm submission count differs")
    _require(slurm.get("automatic_retry") is False, "v16 Slurm retry used")
    _require(slurm.get("second_job_submitted") is False, "v16 Slurm second job submitted")
    _require(slurm.get("requeue") == 0 and slurm.get("restarts") == 0, "v16 Slurm restart differs")
    return cells


def analyze(
    *,
    cpu_record: dict[str, object],
    checkpoint_record: dict[str, object],
    v5_summary: dict[str, object],
    v5_auxiliary: dict[str, object],
    v6_summary: dict[str, object],
    v6_performance: dict[str, object],
    v6_transform: dict[str, object],
    v6_profile_sources: dict[str, object],
    v16_summary: dict[str, object],
    v16_analysis: dict[str, object],
    v16_validation: dict[str, object],
    v16_slurm: dict[str, object],
    archive_verification: dict[str, object],
    input_sha256: dict[str, str],
) -> dict[str, object]:
    _validate_archive_reports(archive_verification)
    _validate_cpu(cpu_record)
    _validate_checkpoint(checkpoint_record)
    _validate_v5(v5_summary, v5_auxiliary)
    reused = _validate_v6(v6_summary, v6_performance, v6_transform, v6_profile_sources)
    requalified = _validate_v16(v16_summary, v16_analysis, v16_validation, v16_slurm)
    _require(requalified == FOCUSED_CELLS, "requalified cell set differs")
    _require(not (reused & requalified), "reused and requalified cells overlap")
    _require(len(reused | requalified) == 16, "cumulative cell coverage differs")
    _require(all(isinstance(value, str) and len(value) == 64 for value in input_sha256.values()), "input identities missing")
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "g1_complete": True,
        "eligible_for_release_integration_review": True,
        "post_release_verification_required": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "new_h100_submission_count": 0,
        "scientific_timestep_executed": False,
        "evidence": {
            "cpu_and_installed_wheel": "PASS",
            "released_checkpoint_matrix": "PASS",
            "h100_auxiliary": "PASS_REUSED_V5",
            "transform_measurement_window": "PASS_REUSED_V6",
            "performance_memory": {
                "passed": True,
                "cell_count": 16,
                "reused_v6_cell_count": 12,
                "requalified_v16_cell_count": 4,
                "requalified_cells": [
                    {"family": family, "grid": grid}
                    for family, grid in sorted(requalified)
                ],
            },
        },
        "historical_classifications_preserved": {
            "v5": v5_summary["classification"],
            "v6": v6_summary["classification"],
            "v16": v16_summary["classification"],
        },
        "archive_verification": archive_verification,
        "input_sha256": dict(sorted(input_sha256.items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--cpu-control", type=Path, required=True)
    parser.add_argument("--checkpoint-control", type=Path, required=True)
    parser.add_argument("--v5-control", type=Path, required=True)
    parser.add_argument("--v5-scratch", type=Path, required=True)
    parser.add_argument("--v6-control", type=Path, required=True)
    parser.add_argument("--v16-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo = arguments.repository_root.expanduser().resolve()
    paths = {
        "cpu_record": repo / "notes/PSSolver_v0_2_0rc5_g1_cpu_closure.json",
        "checkpoint_record": repo / "notes/PSSolver_v0_2_0rc5_g1_checkpoint_matrix.json",
        "v5_summary": arguments.v5_control / "summary.json",
        "v5_auxiliary": arguments.v5_scratch / "results/auxiliary.json",
        "v6_summary": arguments.v6_control / "summary.json",
        "v6_performance": arguments.v6_control / "results/performance_memory_diagnostic.json",
        "v6_transform": arguments.v6_control / "results/transform_window_validation.json",
        "v6_sources": arguments.v6_control / "source_evidence_manifest.json",
        "v16_summary": arguments.v16_control / "summary.json",
        "v16_analysis": arguments.v16_control / "analysis/independent_qualification.json",
        "v16_validation": arguments.v16_control / "analysis/profile_validation.json",
        "v16_slurm": arguments.v16_control / "provenance/slurm_final.json",
    }
    loaded = {name: _json(path) for name, path in paths.items()}
    archive_roots = {
        "cpu": arguments.cpu_control,
        "checkpoint": arguments.checkpoint_control,
        "v5": arguments.v5_control,
        "v6": arguments.v6_control,
        "v16": arguments.v16_control,
    }
    verification = {
        name: verify_archive(
            root,
            expected_entries=int(EXPECTED_ARCHIVES[name]["entries"]),
            expected_manifest_sha256=str(EXPECTED_ARCHIVES[name]["manifest_sha256"]),
        )
        for name, root in archive_roots.items()
    }
    profile_sources = verify_v6_profile_sources(loaded.pop("v6_sources"))
    result = analyze(
        **loaded,
        v6_profile_sources=profile_sources,
        archive_verification=verification,
        input_sha256={name: _sha256(path) for name, path in paths.items()},
    )
    _atomic_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
