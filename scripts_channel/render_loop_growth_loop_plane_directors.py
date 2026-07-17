#!/usr/bin/env python3
"""Render loop-growth panels with directors on the fitted loop plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import render_loop_growth_local_clean_3panel_alt_no_director as base
from nematics3d.classes.q_field_object import InputQ


REPO_ROOT = Path(__file__).resolve().parents[1]


def fit_loop_plane(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords - center, full_matrices=False)
    axis1 = vh[0]
    normal = vh[-1]
    if normal[0] < 0:
        normal = -normal
    axis1 = axis1 - np.dot(axis1, normal) * normal
    axis1 /= np.linalg.norm(axis1)
    return center, normal, axis1


def build_q_object(q5: np.ndarray) -> base.n3d.QFieldObject:
    return base.n3d.QFieldObject(
        inputValue=InputQ(
            Q=q5,
            box_periodic_flag=base.PERIODIC_BOUNDARY,
        ),
        name="loop plane Q",
        is_detect_defects=False,
        is_classify_lines=False,
    )


def render_loop_with_plane_directors(
    output: Path,
    q5: np.ndarray,
    info: base.LoopInfo,
    box: base.RenderBox,
    *,
    plane_size: float,
    plane_spacing: float,
    rod_length: float,
    rod_radius: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = base.local_bounds(box)

    fig = base.PlotFigure(
        is_off_screen=True,
        name=f"loop plane directors step {info.step}",
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
            opacity=0.16,
            sides=8,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )

    plane_center, plane_normal, plane_axis1 = fit_loop_plane(info.coords)
    q_obj = build_q_object(q5)
    q_obj.act_visualize_n_plane(
        figure=fig,
        is_new=False,
        is_extent=False,
        is_defect=False,
        bounds=bounds,
        plane_name=f"loop fitted plane step {info.step}",
        grid_origin=tuple(plane_center),
        grid_normal=tuple(plane_normal),
        grid_axis1=tuple(plane_axis1),
        grid_alignment="center",
        grid_size=plane_size,
        grid_spacing=plane_spacing,
        n_length=rod_length,
        n_radius=rod_radius,
        n_color=(0.25, 0.34, 0.78),
        n_opacity=0.76,
        n_sides=8,
        n_is_scalar_bar=False,
        n_is_reset_camera=False,
        nb_color=(0.25, 0.34, 0.78),
        nb_opacity=0.76,
        nb_is_reset_camera=False,
        nd_color=(0.82, 0.18, 0.14),
        nd_opacity=0.92,
        nd_is_reset_camera=False,
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

    fig.act_view_isometric()
    fig.act_commit(
        focal_point=tuple(box.center),
        distance=82,
        azimuth=18,
        elevation=16,
        roll=-78,
    )
    fig.act_savefig(str(output), scale=2)


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
    parser.add_argument("--plane-size", type=float, default=0.0)
    parser.add_argument("--plane-spacing", type=float, default=2.2)
    parser.add_argument("--rod-length", type=float, default=1.45)
    parser.add_argument("--rod-radius", type=float, default=0.045)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.asarray(args.target_center, dtype=float)
    infos: list[base.LoopInfo] = []
    q_snapshots: list[np.ndarray] = []
    image_paths: list[Path] = []

    for step in args.steps:
        q_path = args.data_dir / f"Q_{step}.npy"
        if not q_path.exists():
            raise FileNotFoundError(q_path)
        q5 = np.load(q_path)
        info = base.choose_loop(step, q5, target)
        target = info.center
        q_snapshots.append(q5)
        infos.append(info)

    box = base.make_render_box(infos, q_snapshots[0].shape[:3], float(args.half_width))
    plane_size = float(args.plane_size)
    if plane_size <= 0:
        plane_size = max(11.0, 2.45 * max(info.radius for info in infos))

    for q5, info in zip(q_snapshots, infos):
        image_path = args.output_dir / f"loop_local_plane_directors_step_{info.step:05d}.png"
        render_loop_with_plane_directors(
            image_path,
            q5,
            info,
            box,
            plane_size=plane_size,
            plane_spacing=float(args.plane_spacing),
            rod_length=float(args.rod_length),
            rod_radius=float(args.rod_radius),
        )
        image_paths.append(image_path)
        plane_center, plane_normal, _ = fit_loop_plane(info.coords)
        print(
            f"step={info.step} image={image_path} radius={info.radius:.3f} "
            f"plane_center={plane_center.round(3).tolist()} "
            f"plane_normal={plane_normal.round(3).tolist()}"
        )

    contact = args.output_dir / "loop_growth_3panel_loop_plane_directors.png"
    base.make_contact_sheet(image_paths, infos, contact)
    print(f"contact_sheet={contact}")


if __name__ == "__main__":
    main()
