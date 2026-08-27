#!/usr/bin/env python3
"""Render every data_control_box snapshot in the beta-overview style.

The Q/director/defect rendering deliberately mirrors
``render_beta_frame.py``.  A screen-space light-bulb indicator reports whether
the activity field is on or off without obscuring the three-dimensional data.
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
import pyvista as pv
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
NEMATICS_SRC = REPO_ROOT.parent / "Nematics3D" / "src"
if NEMATICS_SRC.exists():
    sys.path.insert(0, str(NEMATICS_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nematics3d as n3d
from pssolver.models.active_nematics import q_field_object_from_Q
from nematics3d.classes.visual.plot_figure import OptsFigure, PlotFigure

import render_beta_frame as beta_view


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
    parser.add_argument("--figure-size", type=int, nargs=2, default=(1180, 1120))
    parser.add_argument("--save-scale", type=float, default=1.0)
    parser.add_argument("--line-smooth-window-length", type=int, default=11)
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


def activity_bulb_image(is_on: bool, size: int = 192) -> pv.ImageData:
    """Create a small RGBA light-bulb icon for a PyVista logo widget."""
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    scale = size / 192.0

    def xy(points):
        return tuple(int(round(value * scale)) for value in points)

    if is_on:
        ray_color = (245, 166, 35, 255)
        fill_color = (255, 211, 66, 255)
        outline_color = (214, 132, 18, 255)
        rays = [
            ((96, 9), (96, 34)),
            ((42, 29), (59, 47)),
            ((150, 29), (133, 47)),
            ((20, 78), (48, 78)),
            ((144, 78), (172, 78)),
        ]
        for start, end in rays:
            draw.line((xy(start), xy(end)), fill=ray_color, width=max(3, int(7 * scale)))
    else:
        fill_color = (226, 229, 233, 255)
        outline_color = (112, 119, 128, 255)

    draw.ellipse(
        xy((51, 35, 141, 125)),
        fill=fill_color,
        outline=outline_color,
        width=max(3, int(6 * scale)),
    )
    draw.polygon(
        [xy(point) for point in ((65, 103), (127, 103), (116, 143), (76, 143))],
        fill=fill_color,
    )
    draw.line(
        [xy(point) for point in ((65, 103), (76, 143), (116, 143), (127, 103))],
        fill=outline_color,
        width=max(3, int(6 * scale)),
        joint="curve",
    )
    for y in (143, 153, 163):
        draw.line(xy((76, y, 116, y)), fill=(82, 88, 96, 255), width=max(3, int(6 * scale)))
    draw.arc(xy((81, 160, 111, 180)), 0, 180, fill=(82, 88, 96, 255), width=max(3, int(6 * scale)))
    if is_on:
        draw.ellipse(xy((72, 52, 91, 71)), fill=(255, 250, 205, 210))

    rgba = np.asarray(image, dtype=np.uint8)
    vtk_image = pv.ImageData(dimensions=(size, size, 1))
    vtk_image.point_data["RGBA"] = np.flipud(rgba).reshape((-1, 4))
    vtk_image.set_active_scalars("RGBA")
    return vtk_image


def add_activity_indicator(fig: PlotFigure, amplitude: float) -> None:
    """Add an on/off bulb in the lower-right corner of the frame."""
    is_on = amplitude > 0.5
    icon = activity_bulb_image(is_on)
    fig.pl.add_logo_widget(icon, position=(0.84, 0.035), size=(0.13, 0.13))


def render_one(task: tuple) -> tuple[int, str]:
    (step, frame_number, data_dir_s, output_dir_s, history, figure_size,
     save_scale, line_smooth_window_length, overwrite) = task
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
    q_obj = q_field_object_from_Q(
        q5,
        box_periodic_flag=PERIODIC_BOUNDARY,
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
    add_activity_indicator(fig, amplitude)
    beta_view.add_director_plane(
        fig, q_obj, bounds, center, plane_z=center[2], spacing=2.5,
        length=1.6, radius=0.10, opacity=0.32,
        highlight_defect_directors=False, highlight_radius=0.085, highlight_opacity=0.86,
    )

    plotted = 0
    for line in q_obj.lines:
        coords, smooth = beta_view.smooth_or_raw(
            line,
            density=12.0,
            window_length=int(line_smooth_window_length),
        )
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
    output_dir = (args.output_dir or data_dir / "alpha_beta_movie" / "frames").resolve()
    video = (args.video or output_dir.parent / "control_box_alpha_beta.mp4").resolve()
    steps = available_steps(data_dir, args.start, args.stop, args.stride)
    if not steps:
        raise SystemExit("No matching Q_<step>.npy snapshots found")
    history = load_alpha_amplitudes(data_dir)
    tasks = [
        (step, i, str(data_dir), str(output_dir), history, tuple(args.figure_size),
         args.save_scale, args.line_smooth_window_length, args.overwrite)
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
