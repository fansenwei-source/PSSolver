#!/usr/bin/env python3
"""Fail-closed analyzer for the four frozen rc2 G5 smoke reports."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

if __package__:
    from .run_rc2_g5_installed_smoke import (
        APPLICATIONS,
        FROZEN_INPUT_SHA256,
        FROZEN_SHAPE,
        FROZEN_TOTAL_STEPS,
        KIND,
        VARIANTS,
    )
else:  # Direct installed-wheel qualification script execution.
    from run_rc2_g5_installed_smoke import (
        APPLICATIONS,
        FROZEN_INPUT_SHA256,
        FROZEN_SHAPE,
        FROZEN_TOTAL_STEPS,
        KIND,
        VARIANTS,
    )


PASS_CLASSIFICATION = "PASS_V0_2_0RC2_G5_INSTALLED_SMOKE"
MEMORY_RATIO_MAX = 1.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"report root is not an object: {path}")
    return value


def analyze(reports: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    expected = {(variant, application) for variant in VARIANTS for application in APPLICATIONS}
    _require(set(reports) == expected, "G5 report matrix is incomplete")
    validated: dict[str, object] = {}
    for (variant, application), value in reports.items():
        key = f"{variant}_{application}"
        _require(value.get("kind") == KIND, f"{key} report kind differs")
        _require(value.get("variant") == variant, f"{key} variant differs")
        _require(value.get("application") == application, f"{key} application differs")
        _require(value.get("passed") is True, f"{key} did not pass")
        _require(value.get("all_byte_identical") is True, f"{key} restart differs")
        _require(value.get("all_finite") is True, f"{key} is non-finite")
        contract = value.get("contract")
        _require(isinstance(contract, dict), f"{key} contract is absent")
        _require(contract.get("shape") == list(FROZEN_SHAPE), f"{key} shape differs")
        _require(
            contract.get("initial_q_sha256") == FROZEN_INPUT_SHA256,
            f"{key} input differs",
        )
        _require(
            contract.get("spectral_refresh") == "disabled",
            f"{key} spectral refresh differs",
        )
        for stage in ("continuous", "segment", "resumed"):
            result = value.get(stage)
            _require(isinstance(result, dict), f"{key} {stage} result is absent")
            _require(result.get("complete") is True, f"{key} {stage} is incomplete")
            runtime = result.get("runtime_selection")
            _require(isinstance(runtime, dict), f"{key} runtime identity is absent")
            _require(
                runtime.get("requested") == runtime.get("effective"),
                f"{key} runtime requested/effective differs",
            )
            _require(runtime.get("fallback_used") is False, f"{key} used fallback")
        _require(
            value["continuous"]["final_step"] == FROZEN_TOTAL_STEPS,
            f"{key} continuous final step differs",
        )
        _require(
            value["resumed"]["final_step"] == FROZEN_TOTAL_STEPS,
            f"{key} resumed final step differs",
        )
        environment = value.get("environment")
        _require(isinstance(environment, dict), f"{key} environment is absent")
        _require(environment.get("source_shadow_import") is False, f"{key} source shadow")
        purelib = Path(str(environment.get("purelib", ""))).resolve()
        imported = Path(str(environment.get("pssolver_file", ""))).resolve()
        prefix = Path(str(environment.get("python_prefix", ""))).resolve()
        _require(purelib.is_relative_to(prefix), f"{key} purelib is outside venv")
        _require(imported.is_relative_to(purelib), f"{key} import is outside purelib")
        _require(environment.get("requested_device") == "cuda", f"{key} request differs")
        _require(environment.get("device") == "cuda:0", f"{key} device differs")
        _require(environment.get("tf32_matmul") is False, f"{key} TF32 matmul enabled")
        _require(environment.get("tf32_cudnn") is False, f"{key} TF32 cuDNN enabled")
        validated[key] = {
            "repository": value.get("repository"),
            "memory": value.get("memory"),
            "timing": value.get("timing"),
        }
    ratios: dict[str, object] = {}
    for application in APPLICATIONS:
        baseline = reports[("baseline", application)]["memory"]
        candidate = reports[("candidate", application)]["memory"]
        record = {}
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            denominator = int(baseline[name])
            numerator = int(candidate[name])
            _require(denominator > 0 and numerator > 0, f"{application} {name} invalid")
            ratio = numerator / denominator
            _require(math.isfinite(ratio), f"{application} {name} ratio non-finite")
            _require(ratio <= MEMORY_RATIO_MAX, f"{application} {name} ratio exceeds gate")
            record[f"{name}_ratio"] = ratio
        ratios[application] = record
    return {
        "schema_version": 1,
        "kind": "pssolver_v0_2_0rc2_g5_installed_smoke_analysis",
        "classification": PASS_CLASSIFICATION,
        "memory_ratio_max": MEMORY_RATIO_MAX,
        "ratios": ratios,
        "reports": validated,
        "passed": True,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in VARIANTS:
        for application in APPLICATIONS:
            parser.add_argument(
                f"--{variant}-{application}",
                type=Path,
                required=True,
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = {
        (variant, application): _load(getattr(args, f"{variant}_{application}"))
        for variant in VARIANTS
        for application in APPLICATIONS
    }
    result = analyze(reports)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
