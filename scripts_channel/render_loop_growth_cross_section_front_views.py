#!/usr/bin/env python3
"""Render loop snapshots with front-view director cross sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_loop_growth_local_cross_sections as base


REPO_ROOT = Path(__file__).resolve().parents[1]


def camera_up_from_normal(normal: np.ndarray) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    up = up - np.dot(up, normal) * normal
    if np.linalg.norm(up) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        up = up - np.dot(up, normal) * normal
    return up / np.linalg.norm(up)


def set_front_camera(fig: base.PlotFigure, origin: np.ndarray, normal: np.ndarray, scale: float) -> None:
    normal = normal / np.linalg.norm(normal)
    fig.pl.camera.position = tuple(origin + normal * 30.0)
    fig.pl.camera.focal_point = tuple(origin)
    fig.pl.camera.up = tuple(camera_up_from_normal(normal))
    fig.pl.enable_parallel_projection()
    fig.pl.camera.parallel_scale = scale
    fig.pl.render()


def render_loop_context(
    output: Path,
    info: base.LoopInfo,
    box: base.RenderBox,
    centers: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = base.local_bounds(box)
    fig = base.PlotFigure(
        is_off_screen=True,
        name=f"front-view context step {info.step}",
        opts=base.OptsFigure(
            size=(900, 900),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=tuple(box.center),
        ),
    )
    bounds.act_visualize(
        figure=fig,
        opts=base.n3d.OptsTube(
            radius=0.035,
            color=(0.75, 0.75, 0.75),
            opacity=0.18,
            sides=8,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )
    info.line.act_visualize(
        figure=fig,
        is_wrap=True,
        is_smooth=True,
        opts=base.n3d.OptsTube(
            radius=0.28,
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            sides=32,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
        bounds=bounds,
        clip_mode="mesh",
    )
    base.n3d.PlotSphere(
        centers,
        figure=fig,
        opts=base.OptsSphere(
            radius=0.46,
            color=(0.88, 0.10, 0.10),
            opacity=1.0,
            sides=18,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
        bounds=bounds,
        clip_mode="center",
    )
    fig.act_view_isometric()
    fig.act_commit(
        focal_point=tuple(box.center),
        distance=base.CAMERA_DISTANCE,
        azimuth=base.CAMERA_AZIMUTH,
        elevation=base.CAMERA_ELEVATION,
        roll=base.CAMERA_ROLL,
    )
    fig.act_savefig(str(output), scale=2)


def render_front_section(
    output: Path,
    q_obj: base.n3d.QFieldObject,
    line_index: int,
    u_percent: float,
    bounds: base.n3d.Bounds,
    *,
    step: int,
    section_radius: float,
    section_dr: float,
    section_arc_dist: float,
    rod_length: float,
    rod_radius: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    line = q_obj.lines[line_index]
    origin = base.section_center_from_smooth(line, u_percent)
    normal = base.tangent_at_u(line, u_percent)
    layers = 2
    r_min = max(section_dr, section_radius - section_dr)

    fig = base.PlotFigure(
        is_off_screen=True,
        name=f"front section step {step} u {u_percent:g}",
        opts=base.OptsFigure(
            size=(720, 720),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=tuple(origin),
        ),
    )
    q_obj.act_visualize_n_near_defect(
        u_percent,
        index_line=line_index,
        index_smooth=-1,
        figure=fig,
        is_new=False,
        is_extent=False,
        is_wrap=True,
        bounds=bounds,
        plane_name=f"front section step {step} u {u_percent:g}",
        grid_r_min=r_min,
        grid_layers=layers,
        grid_dr=section_dr,
        grid_arc_dist=section_arc_dist,
        n_length=rod_length,
        n_radius=rod_radius,
        n_color=(0.22, 0.32, 0.78),
        n_opacity=0.92,
        n_sides=10,
        n_is_reset_camera=False,
        nb_color=(0.22, 0.32, 0.78),
        nb_is_reset_camera=False,
        nd_color=(0.85, 0.15, 0.12),
        nd_opacity=1.0,
        nd_is_reset_camera=False,
    )
    base.n3d.PlotSphere(
        np.asarray([origin]),
        figure=fig,
        opts=base.OptsSphere(
            radius=0.18,
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            sides=18,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )
    set_front_camera(fig, origin, normal, scale=section_radius + 2.4)
    fig.act_savefig(str(output), scale=2)


def fit_on_white(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_contact_sheet(context_paths: list[Path], section_paths: list[tuple[Path, Path]], output: Path) -> None:
    top_margin = 72
    col_w = 420
    context_h = 350
    section_h = 210
    width = col_w * len(context_paths)
    height = top_margin + context_h + section_h
    sheet = Image.new("RGB", (width, height), "white")

    for i, context_path in enumerate(context_paths):
        sheet.paste(fit_on_white(context_path, (col_w, context_h)), (i * col_w, top_margin))
        left, right = section_paths[i]
        sheet.paste(fit_on_white(left, (col_w // 2, section_h)), (i * col_w, top_margin + context_h))
        sheet.paste(
            fit_on_white(right, (col_w // 2, section_h)),
            (i * col_w + col_w // 2, top_margin + context_h),
        )

    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 46)
    except Exception:
        font = None
    axis_y = 70
    left_x = col_w // 2
    right_x = width - col_w // 2
    arrow = 20
    draw.line((left_x, axis_y, right_x - arrow, axis_y), fill=(30, 30, 30), width=5)
    draw.polygon(
        [(right_x, axis_y), (right_x - arrow, axis_y - arrow // 2), (right_x - arrow, axis_y + arrow // 2)],
        fill=(30, 30, 30),
    )
    label = "time"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((left_x + right_x - (bbox[2] - bbox[0])) / 2, 4), label, fill=(0, 0, 0), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "loop_growth_local_clean",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=list(base.DEFAULT_STEPS))
    parser.add_argument("--target-center", type=float, nargs=3, default=base.DEFAULT_TARGET_CENTER.tolist())
    parser.add_argument("--half-width", type=float, default=13.0)
    parser.add_argument("--section-radius", type=float, default=2.4)
    parser.add_argument("--section-dr", type=float, default=0.8)
    parser.add_argument("--section-arc-dist", type=float, default=2.1)
    parser.add_argument("--section-rod-length", type=float, default=1.85)
    parser.add_argument("--section-rod-radius", type=float, default=0.075)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.asarray(args.target_center, dtype=float)
    infos: list[base.LoopInfo] = []
    q_snapshots: list[np.ndarray] = []

    for step in args.steps:
        q5 = np.load(args.data_dir / f"Q_{step}.npy")
        info = base.choose_loop(step, q5, target)
        target = info.center
        q_snapshots.append(q5)
        infos.append(info)

    box = base.make_render_box(infos, q_snapshots[0].shape[:3], float(args.half_width))
    bounds = base.local_bounds(box)
    view = base.camera_view_direction(base.CAMERA_AZIMUTH, base.CAMERA_ELEVATION)

    context_paths: list[Path] = []
    section_paths: list[tuple[Path, Path]] = []
    for q5, info in zip(q_snapshots, infos):
        q_obj = base.build_q_object(q5, info.step)
        line_index = base.choose_q_object_line_index(q_obj, info.center)
        line = q_obj.lines[line_index]
        u_pair = base.select_view_parallel_opposite_u_percents(line, view)
        centers = np.asarray([base.section_center_from_smooth(line, u) for u in u_pair], dtype=float)

        context_path = args.output_dir / f"loop_front_context_step_{info.step:05d}.png"
        section_a = args.output_dir / f"loop_front_section_step_{info.step:05d}_a.png"
        section_b = args.output_dir / f"loop_front_section_step_{info.step:05d}_b.png"
        render_loop_context(context_path, info, box, centers)
        render_front_section(
            section_a,
            q_obj,
            line_index,
            u_pair[0],
            bounds,
            step=info.step,
            section_radius=float(args.section_radius),
            section_dr=float(args.section_dr),
            section_arc_dist=float(args.section_arc_dist),
            rod_length=float(args.section_rod_length),
            rod_radius=float(args.section_rod_radius),
        )
        render_front_section(
            section_b,
            q_obj,
            line_index,
            u_pair[1],
            bounds,
            step=info.step,
            section_radius=float(args.section_radius),
            section_dr=float(args.section_dr),
            section_arc_dist=float(args.section_arc_dist),
            rod_length=float(args.section_rod_length),
            rod_radius=float(args.section_rod_radius),
        )
        context_paths.append(context_path)
        section_paths.append((section_a, section_b))
        print(f"step={info.step} u_pair=({u_pair[0]:.2f}, {u_pair[1]:.2f})")

    contact = args.output_dir / "loop_growth_3panel_cross_section_front_views.png"
    make_contact_sheet(context_paths, section_paths, contact)
    print(f"contact_sheet={contact}")


if __name__ == "__main__":
    main()
