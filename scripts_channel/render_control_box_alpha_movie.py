#!/usr/bin/env python3
"""Render every data_control_box snapshot in the beta-overview style plus alpha.

The Q/director/defect rendering deliberately mirrors
``render_nematics3d_beta_overview.py``.  Alpha is drawn first as a faint scalar
surface just behind the director plane, so it remains legible without hiding
the directors or beta-coloured disclination tubes.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NEMATICS_SRC = REPO_ROOT.parent / "Nematics3D" / "src"
if NEMATICS_SRC.exists():
    sys.path.insert(0, str(NEMATICS_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nematics3d as n3d
from nematics3d.classes.q_field_object import InputQ
from nematics3d.classes.visual.plot_figure import OptsFigure, PlotFigure

import render_nematics3d_beta_overview as beta_view


PERIODIC_BOUNDARY = (True, False, False)
ALPHA_ON = 5.0
STEP_RE = re.compile(r"^Q_(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Stride in the sorted saved snapshots.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--figure-size", type=int, nargs=2, default=(984, 1120))
    parser.add_argument("--save-scale", type=float, default=1.0)
    parser.add_argument("--alpha-opacity", type=float, default=0.16)
    parser.add_argument("--alpha-surface-stride", type=int, default=2)
    return parser.parse_args()


def available_steps(data_dir: Path, start: int | None, stop: int | None, stride: int) -> list[int]:
    steps = sorted(
        int(match.group(1))
        for path in data_dir.glob("Q_*.npy")
        if (match := STEP_RE.match(path.name))
    )
    steps = [s for s in steps if (start is None or s >= start) and (stop is None or s <= stop)]
    return steps[::stride]


def load_alpha_amplitudes(data_dir: Path) -> dict[int, float]:
    path = data_dir / "control_history.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            int(row["step"]): float(row["alpha_box_amplitude"])
            for row in csv.DictReader(handle)
        }


def alpha_amplitude_at(step: int, history: dict[int, float]) -> float:
    if not history:
        return ALPHA_ON
    previous = [key for key in history if key <= step]
    return history[max(previous)] if previous else history[min(history)]


def add_alpha_distribution(
    fig: PlotFigure,
    alpha: np.ndarray,
    plane_z: float,
    opacity: float,
    stride: int,
) -> None:
    """Use Nematics3D PlotSurface to show alpha without covering foreground glyphs."""
    z_index = int(np.clip(round(plane_z), 0, alpha.shape[2] - 1))
    x = np.arange(0, alpha.shape[0], stride, dtype=float)
    y = np.arange(0, alpha.shape[1], stride, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # A small negative z offset prevents z-fighting and keeps rods in front.
    zz = np.full_like(xx, float(plane_z) - 0.18)
    scalars = alpha[::stride, ::stride, z_index].ravel().astype(float)
    keep = scalars > 1.0e-6
    if not np.any(keep):
        return
    n3d.PlotSurface(
        np.column_stack((xx.ravel()[keep], yy.ravel()[keep], zz.ravel()[keep])),
        figure=fig,
        name="alpha distribution",
        paint_by="scalars",
        scalars=scalars[keep],
        scalars_cmap="Reds",
        scalars_clim=(0.0, ALPHA_ON),
        opacity=float(opacity),
        is_scalar_bar=False,
        is_reset_camera=False,
    )


def render_one(task: tuple) -> tuple[int, str]:
    (step, frame_number, data_dir_s, output_dir_s, history, figure_size,
     save_scale, alpha_opacity, alpha_stride, overwrite) = task
    data_dir = Path(data_dir_s)
    output = Path(output_dir_s) / f"frame_{frame_number:06d}.png"
    if output.exists() and not overwrite:
        return step, "cached"

    q5 = beta_view.load_q(data_dir, step)
    shape = q5.shape[:3]
    # These are the exact fixed crop and centre used for the uploaded
    # three_snapshot_panel_step_*.png images.
    center = np.array([256.0, 20.0, 20.0])
    bounds = beta_view.make_bounds(
        center, shape, 86.0, x_bounds=(246.0, 266.0),
        y_bounds=(4.0, 36.0), z_bounds=(4.0, 36.0)
    )
    q_obj = n3d.QFieldObject(
        inputValue=InputQ(Q=q5, box_periodic_flag=PERIODIC_BOUNDARY),
        name=f"Q step {step}",
        is_detect_defects=True,
        is_classify_lines=True,
    )
    fig = PlotFigure(
        is_off_screen=True,
        name=f"control-box alpha step {step}",
        opts=OptsFigure(size=tuple(figure_size), bg_color=(1.0, 1.0, 1.0), focal_point=tuple(center)),
    )
    try:
        fig.pl.enable_anti_aliasing("ssaa")
        fig.pl.enable_depth_peeling()
    except Exception:
        pass

    beta_view.add_channel_shell(fig, bounds, stride=2, opacity=0.035)
    amplitude = alpha_amplitude_at(step, history)
    mask = np.load(data_dir / "alpha_box_mask.npy", mmap_mode="r")
    add_alpha_distribution(fig, amplitude * mask, center[2], alpha_opacity, alpha_stride)
    beta_view.add_director_plane(
        fig, q_obj, bounds, center, plane_z=center[2], spacing=2.5,
        length=2.8, radius=0.055, opacity=0.32,
        highlight_defect_directors=False, highlight_radius=0.085, highlight_opacity=0.86,
    )

    plotted = 0
    for line in q_obj.lines:
        coords, smooth = beta_view.smooth_or_raw(line, density=12.0)
        if len(coords) < 4 or not np.any(beta_view.in_bounds(coords, bounds)):
            continue
        # The final three-panel images use Nematics3D's measured beta, which is
        # why the loop contains the full rainbow rather than being almost red.
        beta = beta_view.real_beta_for_line(line, coords, smooth, 181)
        if beta is None:
            beta = beta_view.beta_proxy_from_tangent(coords)
        beta_view.plot_line(
            fig, coords, beta, radius=0.18, sides=72, colormap="turbo",
            opacity=1.0, ambient=0.92, diffuse=0.18, specular=0.02,
            colorbar_position=(0.88, 0.27), colorbar_size=(0.035, 0.42),
            show_scalar_bar=False,
        )
        plotted += 1

    bounds.act_visualize(
        figure=fig,
        opts=n3d.OptsTube(radius=0.025, color=(0.72, 0.72, 0.72), opacity=0.10,
                          sides=8, is_scalar_bar=False, is_reset_camera=False),
    )
    fig.act_view_yz()
    fig.act_commit(focal_point=tuple(center), distance=86.0, azimuth=18.0, elevation=22.0, roll=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.act_savefig(str(output), scale=float(save_scale))
    fig.pl.close()
    return step, f"rendered ({plotted} lines)"


def encode_video(frames_dir: Path, output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "nematics3d_alpha_beta_local_movie" / "frames").resolve()
    video = (args.video or output_dir.parent / "control_box_alpha_beta.mp4").resolve()
    steps = available_steps(data_dir, args.start, args.stop, args.stride)
    if not steps:
        raise SystemExit("No matching Q_<step>.npy snapshots found")
    history = load_alpha_amplitudes(data_dir)
    tasks = [
        (step, i, str(data_dir), str(output_dir), history, tuple(args.figure_size), args.save_scale,
         args.alpha_opacity, args.alpha_surface_stride, args.overwrite)
        for i, step in enumerate(steps)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = map(render_one, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(render_one, tasks)
    for done, (step, status) in enumerate(results, 1):
        print(f"[{done}/{len(steps)}] step={step}: {status}", flush=True)
    if args.workers != 1:
        pool.shutdown()
    encode_video(output_dir, video, args.fps)
    print(f"video={video}")


if __name__ == "__main__":
    main()
