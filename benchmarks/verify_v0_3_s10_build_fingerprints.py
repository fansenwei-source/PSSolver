"""Verify that frozen continuation oracles take their bitwise branch.

Run ``v020`` with the original Nematics3D conda-forge numerical build and
``s08`` with the v0.2.2 release environment.  The helper is qualification
support: it records the branch selected by the public compatibility tests and
does not alter fixtures or runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.v0_2_0_checkpoint_fixtures import (
    numerical_build_fingerprint,
    replay_case,
)


V020_FIXTURES = ROOT / "tests/fixtures/v0_2_0_checkpoints_v1"
V020_ORACLE = ROOT / "tests/oracles/v0_2_0_checkpoint_continuations_v1"
V020_CASES = (
    "functional_periodic",
    "functional_channel",
    "production_periodic",
    "production_channel_complete_stress",
    "production_plane_legacy",
    "production_legacy_channel",
)


def _v020(work_root: Path) -> dict[str, object]:
    reports = {}
    for case in V020_CASES:
        report = replay_case(
            case,
            V020_FIXTURES,
            work_root / case,
            V020_ORACLE,
        )
        continuation = report["continuation"]
        if continuation["mode"] != "bitwise":
            raise AssertionError(f"{case} did not select the bitwise branch")
        if not continuation["matching_numerical_build"]:
            raise AssertionError(f"{case} numerical build does not match")
        reports[case] = {
            "mode": continuation["mode"],
            "matching_numerical_build": True,
            "arrays": sorted(continuation["arrays"]),
        }
    return {"suite": "v020", "cases": reports}


def _s08(work_root: Path) -> dict[str, object]:
    from tests.test_v0_3_s08_historical_cpu_coverage import (
        FIXTURES,
        MANIFEST,
        _channel_runtime,
    )

    fixture = FIXTURES / "rc2_channel_functional_odd"
    runtime = _channel_runtime(work_root / "runtime", fixture)
    target = runtime.checkpoint_bridge.migrate_legacy_checkpoint(
        fixture / "checkpoint",
        work_root / "upgraded",
        source_release_generation="rc2",
    )
    imported = runtime.checkpoint_bridge.import_checkpoint(target)
    controls = {
        "activity": torch.full(
            runtime.control_specs[0].tensor.shape,
            0.01,
            dtype=torch.float64,
        )
    }
    continued = runtime.step(imported.state, controls, 1)
    oracle = fixture / "continuation_oracle"
    expected = (
        torch.from_numpy(np.load(oracle / "q_physical_step_2.npy")),
        torch.from_numpy(np.load(oracle / "q_spectral_step_2.npy")),
    )
    observed = numerical_build_fingerprint()
    reference = MANIFEST["generation_environment"][
        "numerical_build_fingerprint"
    ]
    matching = observed == reference
    arrays = {
        name: torch.equal(actual, wanted)
        for name, actual, wanted in zip(
            ("q_physical", "q_spectral"),
            continued,
            expected,
            strict=True,
        )
    }
    if not matching or not all(arrays.values()):
        raise AssertionError("S-08 rc2 continuation did not select bitwise")
    return {
        "suite": "s08",
        "case": "rc2_channel_functional_odd",
        "mode": "bitwise",
        "matching_numerical_build": matching,
        "arrays_bitwise_equal": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("v020", "s08"), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.work_root.exists():
        raise FileExistsError(args.work_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.work_root.mkdir(parents=True)
    try:
        result = (
            _v020(args.work_root)
            if args.suite == "v020"
            else _s08(args.work_root)
        )
        result["environment"] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "python": sys.version,
            "executable": sys.executable,
            "fingerprint": numerical_build_fingerprint(),
        }
        if result["environment"]["cuda_available"]:
            raise AssertionError("qualification must run with CUDA hidden")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
        json.loads(args.output.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(args.work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
