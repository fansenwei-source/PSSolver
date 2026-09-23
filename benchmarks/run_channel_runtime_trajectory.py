#!/usr/bin/env python3
"""Run one auditable Channel trajectory through the P7.5 package application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pssolver.applications.channel_active_nematics import run_channel_active_nematics
from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.configuration.channel_active_nematics_declarations import ChannelRuntimePath


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-path", choices=tuple(item.value for item in ChannelRuntimePath), required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--activity", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--diagnostic-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = ChannelActiveNematicRunSpec(
        shape=args.shape, lengths=args.lengths, dt=args.dt, steps=args.steps,
        save_interval=args.save_interval,
        diagnostic_interval=args.diagnostic_interval,
        checkpoint_interval=args.checkpoint_interval,
        restart_from=args.restart_from,
        seed=args.seed, activity=args.activity, device=args.device,
        generated_output_directory=args.output,
        runtime_path=args.runtime_path,
    )
    result = run_channel_active_nematics(spec)
    print(json.dumps({
        "runtime_path": args.runtime_path,
        "start_step": result.start_step,
        "final_step": result.final_step,
        "saved_steps": list(result.saved_steps),
        "checkpoint_steps": list(result.checkpoint_steps),
        "output": str(args.output.expanduser().resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
