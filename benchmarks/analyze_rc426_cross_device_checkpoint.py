#!/usr/bin/env python3
"""Fail-closed analyzer for the RC4.2.6 X01--X14 matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from benchmarks.run_rc426_cross_device_checkpoint import (
    COMPLETE_SCHEMA,
    MATRIX,
    PASS_CLASSIFICATION,
    PLAN_SCHEMA,
    SCHEMA,
)


ANALYSIS_SCHEMA = "pssolver.rc4_2_6.cross_device_checkpoint_analysis.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _require(
            json.loads(path.read_text(encoding="utf-8")) == value,
            "analysis JSON round-trip differs",
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_cells() -> list[dict[str, object]]:
    return [cell.to_metadata() for cell in MATRIX]


def _validate_plan(plan: dict[str, Any]) -> None:
    _require(plan.get("schema") == PLAN_SCHEMA, "matrix plan schema differs")
    _require(plan.get("count") == 14, "matrix plan count differs")
    _require(plan.get("cells") == _expected_cells(), "matrix plan differs")
    boundary = plan.get("claim_boundary")
    _require(isinstance(boundary, dict), "claim boundary is missing")
    _require(boundary.get("exact_serialized_restore") is True, "restore claim missing")
    _require(boundary.get("finite_post_restore_step") is True, "smoke claim missing")
    for forbidden in (
        "post_restore_cross_device_byte_identity",
        "performance_or_memory",
        "long_run",
    ):
        _require(boundary.get(forbidden) is False, f"scope expanded: {forbidden}")


def _validate_cell(report: dict[str, Any], expected: dict[str, object]) -> None:
    _require(report.get("schema") == SCHEMA, f"{expected['id']} schema differs")
    _require(report.get("cell") == expected, f"{expected['id']} identity differs")
    _require(report.get("passed") is True, f"{expected['id']} did not pass")
    checkpoint = report.get("checkpoint")
    _require(isinstance(checkpoint, dict), f"{expected['id']} checkpoint missing")
    for key in (
        "ordinary_non_symlink_directory",
        "source_tree_unchanged",
        "compatibility_identity_equal",
    ):
        _require(checkpoint.get(key) is True, f"{expected['id']} {key} failed")
    source_identity = checkpoint.get("compatibility_identity_source")
    target_identity = checkpoint.get("compatibility_identity_target")
    for name, value in (
        ("source identity", source_identity),
        ("target identity", target_identity),
    ):
        _require(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef"),
            f"{expected['id']} {name} is invalid",
        )
    negative = checkpoint.get("negative_integrity_gate")
    _require(isinstance(negative, dict), f"{expected['id']} negative gate missing")
    _require(negative.get("rejected") is True, f"{expected['id']} tamper accepted")
    _require(
        negative.get("integrity_guard_reached") is True,
        f"{expected['id']} wrong negative guard reached",
    )
    _require(
        negative.get("target_unchanged") is True,
        f"{expected['id']} target mutated before rejection",
    )
    progress = report.get("progress")
    _require(isinstance(progress, dict), f"{expected['id']} progress missing")
    _require(progress.get("expected_completed_steps") == 1, "source steps differ")
    _require(progress.get("restored_completed_steps") == 1, "restore steps differ")
    _require(progress.get("exact") is True, f"{expected['id']} progress differs")
    _require(
        progress.get("serialized_equal") is True,
        f"{expected['id']} serialized progress differs",
    )
    state = report.get("state")
    _require(isinstance(state, dict), f"{expected['id']} state missing")
    _require(
        state.get("serialized_payload_equal") is True,
        f"{expected['id']} serialized tensors differ",
    )
    _require(
        state.get("persistent_backend_state_equal") is True,
        f"{expected['id']} backend state differs",
    )
    _require(state.get("restored_finite") is True, f"{expected['id']} restore nonfinite")
    _require(
        state.get("post_restore_step_finite") is True,
        f"{expected['id']} post-restore step nonfinite",
    )
    source_device = "cuda:0" if expected["source"] == "cuda" else "cpu"
    target_device = "cuda:0" if expected["target"] == "cuda" else "cpu"
    _require(state.get("source_devices") == [source_device], "source device differs")
    _require(state.get("target_devices") == [target_device], "target device differs")
    runtime = report.get("runtime")
    _require(isinstance(runtime, dict), f"{expected['id']} runtime missing")
    _require(runtime.get("fallback_used") is False, f"{expected['id']} fallback")


def analyze(
    plan: dict[str, Any],
    reports: Iterable[dict[str, Any]],
    completion: dict[str, Any],
) -> dict[str, object]:
    _validate_plan(plan)
    values = list(reports)
    expected = _expected_cells()
    _require(len(values) == len(expected), "matrix report count differs")
    for report, cell in zip(values, expected, strict=True):
        _validate_cell(report, cell)
    ids = [value["cell"]["id"] for value in values]
    expected_ids = [value["id"] for value in expected]
    _require(ids == expected_ids, "matrix cell order differs")
    _require(len(set(ids)) == 14, "matrix cell IDs are duplicated")
    _require(completion.get("schema") == COMPLETE_SCHEMA, "completion schema differs")
    _require(
        completion.get("classification") == PASS_CLASSIFICATION,
        "completion classification differs",
    )
    _require(completion.get("qualification_complete") is True, "matrix incomplete")
    _require(completion.get("cell_count") == 14, "completion count differs")
    _require(completion.get("ordered_cell_ids") == expected_ids, "completion order differs")
    return {
        "schema": ANALYSIS_SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "rc4_2_6_complete": True,
        "rc4_2_complete": True,
        "eligible_for_rc4_3_planning": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "matrix": {
            "expected": 14,
            "passed": 14,
            "ordered_cell_ids": expected_ids,
            "directions": {"cpu_to_cuda": 7, "cuda_to_cpu": 7},
        },
        "claim_boundary": {
            "checkpoint_materialization_exact": True,
            "post_restore_step_finite": True,
            "post_restore_cross_device_byte_identity_claimed": False,
            "performance_or_memory_claimed": False,
            "scientific_long_run_claimed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(arguments.matrix_root).expanduser().resolve()
    reports = [_load(root / "reports" / f"{cell.id}.json") for cell in MATRIX]
    result = analyze(
        _load(root / "plan.json"),
        reports,
        _load(root / "MATRIX_COMPLETE.json"),
    )
    _atomic_json(Path(arguments.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
