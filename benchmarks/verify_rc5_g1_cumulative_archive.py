#!/usr/bin/env python3
"""Atomically verify one or more rc5 G-1 cumulative-closure archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.analyze_rc5_g1_cumulative_closure import (
    EXPECTED_ARCHIVES,
    _atomic_json,
    verify_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        nargs=2,
        metavar=("NAME", "ROOT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    reports: dict[str, object] = {}
    for name, root_text in arguments.archive:
        if name in reports:
            raise ValueError(f"duplicate archive name: {name}")
        if name not in EXPECTED_ARCHIVES:
            raise ValueError(f"unknown archive name: {name}")
        expected = EXPECTED_ARCHIVES[name]
        reports[name] = verify_archive(
            Path(root_text),
            expected_entries=int(expected["entries"]),
            expected_manifest_sha256=str(expected["manifest_sha256"]),
        )
    _atomic_json(arguments.output, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
