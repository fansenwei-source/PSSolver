#!/usr/bin/env python3
"""Fail-closed RC4.3.7 analyzer for correct-layer CUDA evidence."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Iterable

from benchmarks.analyze_rc437_lifting_h100 import (
    PASS,
    _atomic_json,
    _load,
    analyze as _analyze_v1,
)
from benchmarks.qualify_rc437_lifting_cuda import SCHEMA as CUDA_SCHEMA_V1
from benchmarks.qualify_rc437_lifting_cuda_recovery import (
    MEASUREMENT_LAYER,
    SCHEMA as CUDA_SCHEMA,
)
from pssolver.models.active_nematics import Q_COMPONENTS


SCHEMA = "pssolver.rc4_3_7.lifting_h100_analysis.v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_correct_layer(report: dict[str, Any]) -> None:
    _require(report.get("schema") == CUDA_SCHEMA, "CUDA recovery schema differs")
    fresh = report.get("fresh_conditioning", {})
    _require(
        fresh.get("measurement_layer") == MEASUREMENT_LAYER,
        "conditioning measurement layer differs",
    )
    _require(
        fresh.get("interior_byte_identical_to_unconditioned") is True,
        "direct conditioning interior changed",
    )
    _require(
        fresh.get("transition_region_changed") is True,
        "direct conditioning transition did not change",
    )
    _require(
        float(fresh.get("cross_device_relative_l2", math.inf)) <= 1.0e-15,
        "CPU/CUDA direct conditioning differs",
    )
    cuda = fresh.get("cuda", {})
    components = cuda.get("components", {})
    _require(
        set(components) == set(Q_COMPONENTS),
        "direct conditioning component set differs",
    )
    _require(
        all(
            value.get("interior_byte_identical_to_unconditioned") is True
            for value in components.values()
        ),
        "direct conditioning component interior changed",
    )
    diagnostic = fresh.get("post_projection_diagnostic", {})
    _require(
        diagnostic.get("qualification_gate") is False,
        "post-projection diagnostic became a qualification gate",
    )
    _require(
        diagnostic.get("measurement_layer")
        == "adapter_fields_spatial_after_global_truncated_spectral_projection",
        "post-projection diagnostic layer differs",
    )


def analyze(
    baseline_reports: Iterable[dict[str, Any]],
    candidate_reports: Iterable[dict[str, Any]],
    cuda_report: dict[str, Any],
    *,
    baseline_commit: str,
    candidate_commit: str,
    baseline_root: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    _validate_correct_layer(cuda_report)
    legacy = copy.deepcopy(cuda_report)
    legacy["schema"] = CUDA_SCHEMA_V1
    result = _analyze_v1(
        baseline_reports,
        candidate_reports,
        legacy,
        baseline_commit=baseline_commit,
        candidate_commit=candidate_commit,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )
    result["schema"] = SCHEMA
    result["cuda_correctness"] = {
        "classification": cuda_report["classification"],
        "internal_lifting_infrastructure_only": True,
        "measurement_layer": MEASUREMENT_LAYER,
        "post_projection_role": "diagnostic_only",
    }
    result["qualification_recovery"] = {
        "reason": "correct the interior-identity measurement layer",
        "runtime_source_changed": False,
        "scientific_contract_relaxed": False,
    }
    _require(result.get("classification") == PASS, "aggregate classification differs")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", action="append", type=Path, required=True)
    parser.add_argument(
        "--candidate-profile",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        (_load(path) for path in args.baseline_profile),
        (_load(path) for path in args.candidate_profile),
        _load(args.cuda_report),
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
