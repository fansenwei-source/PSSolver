#!/usr/bin/env python3
"""Fail-closed analyzer for the rc5 G-1 corrected measurement window."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.analyze_rc5_g1_focused_performance_h100 import (
    _analyze_with_plan,
)
from benchmarks.plan_rc5_g1_measurement_window_recovery import build_plan


SCHEMA = "pssolver.rc5.g1.measurement_window_analysis.v1"
PASS_CLASSIFICATION = "PASS_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_EVIDENCE"


def analyze(profile_root: Path) -> dict[str, object]:
    result = _analyze_with_plan(
        profile_root,
        build_plan(),
        analysis_schema=SCHEMA,
        pass_classification=PASS_CLASSIFICATION,
    )
    result.update(
        {
            "measurement_window_corrected": True,
            "post_flush_rewarm_steps": 3,
            "formal_raw_samples_per_profile": 60,
            "diagnostic_replay_is_non_adjudicating": True,
            "v7_failure_reclassified": False,
        }
    )
    return result


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
            raise RuntimeError("analysis JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
