#!/usr/bin/env python3
"""Validate DAL soft-core and centerline coordinates before optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pssolver.control import (
    conservative_block_average,
    core_centerline_geometry,
    periodic_translate_axis,
    soft_core_center,
)


LENGTHS = (128.0, 10.0, 10.0)
S_BULK = 0.5393446629166317


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial",
        type=Path,
        default=REPO_ROOT / "data_optimal_control_loop_translation/Q_initial_pure_loop.npy",
    )
    parser.add_argument("--displacement", type=float, default=-2.0)
    parser.add_argument("--x-strides", type=int, nargs="+", default=(8, 4, 2))
    parser.add_argument("--yz-stride", type=int, default=2)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.55, 0.65, 0.75)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_dal_core_geometry_validation",
    )
    return parser.parse_args()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def compact_geometry(geometry):
    return {name: value for name, value in geometry.items() if name != "centerline"}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.initial)
    results = []
    for x_stride in args.x_strides:
        factors = (x_stride, args.yz_stride, args.yz_stride)
        initial = conservative_block_average(source, factors)
        target = periodic_translate_axis(
            initial, args.displacement, LENGTHS[0], axis=0
        )
        stride_result = {
            "factors": factors,
            "shape": initial.shape[:3],
            "spacing": [
                length / size for length, size in zip(LENGTHS, initial.shape[:3])
            ],
            "threshold_sweep": [],
        }
        for threshold in args.thresholds:
            initial_soft = soft_core_center(
                initial,
                LENGTHS,
                S_bulk=S_BULK,
                threshold_fraction=threshold,
            )
            target_soft = soft_core_center(
                target,
                LENGTHS,
                S_bulk=S_BULK,
                threshold_fraction=threshold,
            )
            initial_line = core_centerline_geometry(
                initial,
                LENGTHS,
                S_bulk=S_BULK,
                threshold_fraction=threshold,
            )
            target_line = core_centerline_geometry(
                target,
                LENGTHS,
                S_bulk=S_BULK,
                threshold_fraction=threshold,
            )
            center_delta = np.asarray(target_line["center"]) - np.asarray(
                initial_line["center"]
            )
            center_delta[0] = (
                center_delta[0] + 0.5 * LENGTHS[0]
            ) % LENGTHS[0] - 0.5 * LENGTHS[0]
            stride_result["threshold_sweep"].append(
                {
                    "threshold_fraction": threshold,
                    "initial_soft": initial_soft,
                    "target_soft": target_soft,
                    "initial_line": compact_geometry(initial_line),
                    "target_line": compact_geometry(target_line),
                    "measured_displacement": center_delta,
                    "displacement_error": center_delta
                    - np.asarray((args.displacement, 0.0, 0.0)),
                    "line_length_relative_change": (
                        target_line["line_length"] / initial_line["line_length"] - 1.0
                    ),
                    "radius_relative_change": (
                        target_line["radius_mean"] / initial_line["radius_mean"] - 1.0
                    ),
                }
            )
        results.append(stride_result)
    summary = {
        "initial": str(args.initial),
        "lengths": LENGTHS,
        "S_bulk": S_BULK,
        "prescribed_displacement": (args.displacement, 0.0, 0.0),
        "results": results,
    }
    output_path = args.output / "core_geometry_validation.json"
    output_path.write_text(json.dumps(json_value(summary), indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
