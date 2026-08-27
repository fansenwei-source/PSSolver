#!/usr/bin/env python3
"""Compare all saved frames from the three 100k-step dealiasing runs."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = {
    "two_thirds": ROOT / "data_free_slip_dealias_two_thirds_dt0p001_100000_save1000",
    "cubic_half": ROOT / "data_free_slip_dealias_cubic_half_dt0p001_100000_save1000",
    "none": ROOT / "data_free_slip_dealias_control_none_dt0p001_100000_save1000",
}
NONE_RESUME = ROOT / "data_free_slip_dealias_control_none_dt0p001_resume96000_to100000"


def path_for(mode, field, step):
    if mode == "none" and step > 96000:
        return NONE_RESUME / f"{field}_{step - 96000}.npy"
    return RUNS[mode] / f"{field}_{step}.npy"


def stats(path, components=False):
    array = np.load(path, mmap_mode="r")
    count = 0
    square_sum = 0.0
    max_abs = 0.0
    finite = True
    component_sum = np.zeros(array.shape[-1]) if components else None
    component_count = 0
    for start in range(0, array.shape[0], 16):
        chunk = np.asarray(array[start:start + 16], dtype=np.float64)
        finite &= bool(np.isfinite(chunk).all())
        count += chunk.size
        square_sum += float(np.square(chunk).sum())
        max_abs = max(max_abs, float(np.max(np.abs(chunk))))
        if components:
            component_sum += chunk.sum(axis=tuple(range(chunk.ndim - 1)))
            component_count += int(np.prod(chunk.shape[:-1]))
    result = {"finite": finite, "rms": (square_sum / count) ** 0.5, "max_abs": max_abs}
    if components:
        result["means"] = component_sum / component_count
    return result


def main():
    rows = []
    for mode in RUNS:
        for step in range(0, 100001, 1000):
            q = stats(path_for(mode, "Q", step))
            u = stats(path_for(mode, "u", step), components=True)
            rows.append({
                "mode": mode, "step": step, "time": step * 0.001,
                "finite": q["finite"] and u["finite"],
                "q_rms": q["rms"], "q_max_abs": q["max_abs"],
                "u_rms": u["rms"], "u_max_abs": u["max_abs"],
                "ux_mean": u["means"][0], "uy_mean": u["means"][1],
                "uz_mean": u["means"][2],
            })
        print(f"finished {mode}")

    timeseries = ROOT / "dealias_dt0p001_100000_timeseries.csv"
    with timeseries.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for mode in RUNS:
        selected = [r for r in rows if r["mode"] == mode]
        late = [r for r in selected if r["step"] >= 50000]
        final = selected[-1]
        summaries.append({
            "mode": mode, "frames": len(selected),
            "all_finite": all(r["finite"] for r in selected),
            "final_q_rms": final["q_rms"], "final_u_rms": final["u_rms"],
            "final_u_max_abs": final["u_max_abs"],
            "late_q_rms_mean": np.mean([r["q_rms"] for r in late]),
            "late_q_rms_std": np.std([r["q_rms"] for r in late]),
            "late_u_rms_mean": np.mean([r["u_rms"] for r in late]),
            "late_u_rms_std": np.std([r["u_rms"] for r in late]),
            "late_u_rms_min": np.min([r["u_rms"] for r in late]),
            "late_u_rms_max": np.max([r["u_rms"] for r in late]),
            "late_u_max_mean": np.mean([r["u_max_abs"] for r in late]),
            "late_u_max_peak": np.max([r["u_max_abs"] for r in late]),
            "max_tangential_mean": max(
                max(abs(r["ux_mean"]), abs(r["uy_mean"])) for r in selected
            ),
        })

    summary = ROOT / "dealias_dt0p001_100000_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(*summaries, sep="\n")


if __name__ == "__main__":
    main()
