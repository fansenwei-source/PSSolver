#!/usr/bin/env python3
"""Render a Nematics3D-style beta overview from one channel snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
]
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from nematics3d.classes.bounds import OptsBounds
from nematics3d.classes.q_field_object import InputQ
from nematics3d.classes.visual.plot_figure import OptsFigure, PlotFigure
from nematics3d.classes.visual.plot_rod import OptsRod


PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_STEP = 7000
DEFAULT_CENTER = np.array([256.0, 20.0, 20.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data_control_box"
        / "nematics3d_beta_overview"
        / f"beta_overview_step_{DEFAULT_STEP:05d}.png",
    )
    parser.add_argument("--center", type=float, nargs=3, default=DEFAULT_CENTER.tolist())
    parser.add_argument("--figure-size", type=int, nargs=2, default=(1900, 1250))
    parser.add_argument("--save-scale", type=float, default=3.0)
    parser.add_argument("--x-half-width", type=float, default=86.0)
    parser.add_argument("--focus-loop", action="store_true")
    parser.add_argument("--local-half-widths", type=float, nargs=3, default=(12.0, 13.0, 13.0))
    parser.add_argument("--local-x-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--local-y-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--local-z-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--director-spacing", type=float, default=5.0)
    parser.add_argument("--director-length", type=float, default=2.8)
    parser.add_argument("--director-radius", type=float, default=0.055)
    parser.add_argument("--director-opacity", type=float, default=0.32)
    parser.add_argument("--director-plane-z", type=float, default=None)
    parser.add_argument("--highlight-defect-directors", action="store_true")
    parser.add_argument("--highlight-director-radius", type=float, default=0.085)
    parser.add_argument("--highlight-director-opacity", type=float, default=0.86)
    parser.add_argument("--line-radius", type=float, default=0.18)
    parser.add_argument("--line-sides", type=int, default=72)
    parser.add_argument("--line-smooth-density", type=float, default=12.0)
    parser.add_argument("--beta-samples", type=int, default=181)
    parser.add_argument("--line-colormap", default="turbo")
    parser.add_argument("--line-opacity", type=float, default=1.0)
    parser.add_argument("--line-ambient", type=float, default=0.92)
    parser.add_argument("--line-diffuse", type=float, default=0.18)
    parser.add_argument("--line-specular", type=float, default=0.02)
    parser.add_argument("--surface-stride", type=int, default=2)
    parser.add_argument("--min-line-points", type=int, default=4)
    parser.add_argument("--bounds-radius", type=float, default=0.025)
    parser.add_argument("--bounds-opacity", type=float, default=0.10)
    parser.add_argument("--shell-opacity", type=float, default=0.035)
    parser.add_argument("--hide-bounds", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=170.0)
    parser.add_argument("--camera-azimuth", type=float, default=18.0)
    parser.add_argument("--camera-elevation", type=float, default=22.0)
    parser.add_argument("--camera-roll", type=float, default=-71.0)
    parser.add_argument("--camera-window-center", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--colorbar-position", type=float, nargs=2, default=(0.88, 0.27))
    parser.add_argument("--colorbar-size", type=float, nargs=2, default=(0.035, 0.42))
    parser.add_argument("--hide-colorbar", action="store_true")
    parser.add_argument("--axes-viewport", type=float, nargs=4, default=(0.03, 0.03, 0.23, 0.23))
    parser.add_argument("--axes-line-width", type=float, default=2.0)
    parser.add_argument("--axes-label-font-size", type=int, default=22)
    parser.add_argument("--axes-label-box", type=float, nargs=2, default=(0.25, 0.10))
    parser.add_argument("--real-beta", action="store_true")
    parser.add_argument("--show-axes", action="store_true")
    return parser.parse_args()


def load_q(data_dir: Path, step: int) -> np.ndarray:
    path = data_dir / f"Q_{step}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    q5 = np.load(path)
    if q5.ndim != 4 or q5.shape[-1] != 5:
        raise ValueError(f"{path} has shape {q5.shape}; expected (Nx, Ny, Nz, 5)")
    return np.asarray(q5, dtype=np.float64)


def make_bounds(
    center: np.ndarray,
    shape: tuple[int, int, int],
    x_half_width: float,
    y_half_width: float | None = None,
    z_half_width: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> n3d.Bounds:
    nx, ny, nz = shape
    if x_bounds is not None:
        x0 = max(0.0, float(min(x_bounds)))
        x1 = min(float(nx), float(max(x_bounds)))
    else:
        x0 = max(0.0, float(center[0] - x_half_width))
        x1 = min(float(nx), float(center[0] + x_half_width))
    if y_bounds is not None:
        y0 = max(0.0, float(min(y_bounds)))
        y1 = min(float(ny), float(max(y_bounds)))
    elif y_half_width is None:
        y0, y1 = 0.0, float(ny)
    else:
        y0 = max(0.0, float(center[1] - y_half_width))
        y1 = min(float(ny), float(center[1] + y_half_width))
    if z_bounds is not None:
        z0 = max(0.0, float(min(z_bounds)))
        z1 = min(float(nz), float(max(z_bounds)))
    elif z_half_width is None:
        z0, z1 = 0.0, float(nz)
    else:
        z0 = max(0.0, float(center[2] - z_half_width))
        z1 = min(float(nz), float(center[2] + z_half_width))
    return n3d.Bounds(
        opts=OptsBounds(
            origin=(x0, y0, z0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=x1 - x0,
            length2=y1 - y0,
            length3=z1 - z0,
            alignment="min_corner",
        )
    )


def smooth_or_raw(line: object, density: float = 1.0) -> tuple[np.ndarray, object | None]:
    if getattr(line, "calc_defect_num", 0) >= 5:
        try:
            smooth = line.act_smooth(
                window_length=5,
                num_out_ratio=float(density),
                min_line_length=5,
                is_window_warning=False,
            )
            coords = np.asarray(smooth.result, dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 3:
                return coords, smooth
        except Exception:
            pass
    return np.asarray(line.calc_defect_coords, dtype=float), None


def in_bounds(coords: np.ndarray, bounds: n3d.Bounds, margin: float = 2.0) -> np.ndarray:
    x0, y0, z0 = np.asarray(bounds.opts.origin, dtype=float) - margin
    x1 = x0 + float(bounds.opts.length1) + 2.0 * margin
    y1 = y0 + float(bounds.opts.length2) + 2.0 * margin
    z1 = z0 + float(bounds.opts.length3) + 2.0 * margin
    return (
        (coords[:, 0] >= x0)
        & (coords[:, 0] <= x1)
        & (coords[:, 1] >= y0)
        & (coords[:, 1] <= y1)
        & (coords[:, 2] >= z0)
        & (coords[:, 2] <= z1)
    )


def beta_proxy_from_tangent(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.zeros(len(coords), dtype=float)
    tangent = np.gradient(coords, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm == 0.0] = 1.0
    tangent = tangent / norm[:, None]
    cos_beta = np.clip(np.abs(tangent[:, 0]), 0.0, 1.0)
    return np.arccos(cos_beta)


def real_beta_for_line(
    line: object,
    coords: np.ndarray,
    beta_source: object | None,
    samples_count: int,
) -> np.ndarray | None:
    if len(coords) < 5:
        return None
    if beta_source is None:
        beta_source = line
    if not hasattr(beta_source, "act_calc_omega"):
        try:
            beta_source = line.act_smooth(
                window_length=5,
                num_out_ratio=max(1.0, len(coords) / max(1, getattr(line, "calc_defect_num", len(coords)))),
                min_line_length=5,
                is_window_warning=False,
            )
        except Exception:
            return None
    samples = np.linspace(0.0, 100.0, max(5, int(samples_count)), endpoint=False)
    values = []
    positions = []
    for sample in samples:
        try:
            result = beta_source.act_calc_omega(
                float(sample),
                opts_grid=n3d.OptsPlaneGridPolar(layers=6, dr=0.45, arc_dist=0.9),
            )
        except Exception:
            return None
        beta = result.get("beta", np.nan)
        if np.isfinite(beta):
            values.append(np.deg2rad(float(beta)))
            positions.append(float(sample))
    if len(values) < 2:
        return None
    u = np.linspace(0.0, 100.0, len(coords))
    return np.interp(u, np.asarray(positions), np.asarray(values))


def plot_line(
    fig: PlotFigure,
    coords: np.ndarray,
    beta: np.ndarray,
    *,
    radius: float,
    sides: int,
    colormap: str,
    opacity: float,
    ambient: float,
    diffuse: float,
    specular: float,
    colorbar_position: tuple[float, float],
    colorbar_size: tuple[float, float],
    show_scalar_bar: bool,
) -> object:
    tube = n3d.PlotTube(
        coords,
        figure=fig,
        paint_by="scalars",
        scalars=np.clip(beta, 0.0, np.pi / 2.0),
        scalars_cmap=colormap,
        scalars_clim=(0.0, np.pi / 2.0),
        scalar_bar_title="",
        radius=radius,
        opacity=opacity,
        sides=sides,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        is_scalar_bar=False,
        is_reset_camera=False,
    )
    if show_scalar_bar:
        bar_x, bar_y = colorbar_position
        bar_width, bar_height = colorbar_size
        fig.pl.add_scalar_bar(
            title="",
            mapper=tube.entity_actor.mapper,
            n_labels=0,
            position_x=bar_x,
            position_y=bar_y,
            width=bar_width,
            height=bar_height,
            vertical=True,
            label_font_size=1,
            color="black",
            outline=False,
            render=False,
        )
        font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        label_x = bar_x + bar_width + 0.012
        fig.pl.add_text(
            "β",
            position=(bar_x, bar_y + bar_height + 0.025),
            font_size=22,
            color="black",
            viewport=True,
            font_file=font_file,
        )
        fig.pl.add_text(
            "π/2",
            position=(label_x, bar_y + bar_height - 0.010),
            font_size=22,
            color="black",
            viewport=True,
            font_file=font_file,
        )
        fig.pl.add_text(
            "0",
            position=(label_x, bar_y - 0.010),
            font_size=22,
            color="black",
            viewport=True,
            font_file=font_file,
        )
    return tube


def add_channel_shell(
    fig: PlotFigure,
    bounds: n3d.Bounds,
    stride: int,
    opacity: float,
) -> None:
    x0, y0, z0 = np.asarray(bounds.opts.origin, dtype=float)
    x1 = x0 + float(bounds.opts.length1)
    y1 = y0 + float(bounds.opts.length2)
    z1 = z0 + float(bounds.opts.length3)

    def axis_points(start: float, stop: float) -> np.ndarray:
        count = max(2, int(np.ceil((stop - start) / max(1, stride))) + 1)
        return np.linspace(start, stop, count, dtype=float)

    x = axis_points(x0, x1)
    y = axis_points(y0, y1)
    z = axis_points(z0, z1)
    faces = [
        ("x-min shell", y, z, 0, x0),
        ("x-max shell", y, z, 0, x1),
        ("y-min shell", x, z, 1, y0),
        ("y-max shell", x, z, 1, y1),
        ("z-min shell", x, y, 2, z0),
        ("z-max shell", x, y, 2, z1),
    ]
    for name, axis_a, axis_b, fixed_axis, fixed_value in faces:
        aa, bb = np.meshgrid(axis_a, axis_b, indexing="ij")
        coords = np.zeros((aa.size, 3), dtype=float)
        if fixed_axis == 0:
            coords[:, 0] = fixed_value
            coords[:, 1] = aa.ravel()
            coords[:, 2] = bb.ravel()
        elif fixed_axis == 1:
            coords[:, 0] = aa.ravel()
            coords[:, 1] = fixed_value
            coords[:, 2] = bb.ravel()
        else:
            coords[:, 0] = aa.ravel()
            coords[:, 1] = bb.ravel()
            coords[:, 2] = fixed_value
        n3d.PlotSurface(
            coords,
            figure=fig,
            name=name,
            color=(0.70, 0.73, 0.70),
            opacity=opacity,
            is_scalar_bar=False,
            is_reset_camera=False,
        )


def add_director_plane(
    fig: PlotFigure,
    q_obj: n3d.QFieldObject,
    bounds: n3d.Bounds,
    center: np.ndarray,
    plane_z: float,
    spacing: float,
    length: float,
    radius: float,
    opacity: float,
    highlight_defect_directors: bool,
    highlight_radius: float,
    highlight_opacity: float,
) -> None:
    origin = np.asarray(bounds.opts.origin, dtype=float)
    nd_opacity = highlight_opacity if highlight_defect_directors else 0.24
    nd_radius = highlight_radius if highlight_defect_directors else radius
    q_obj.act_visualize_n_plane(
        figure=fig,
        is_new=False,
        is_extent=False,
        is_defect=False,
        bounds=bounds,
        plane_name="middle z director plane",
        grid_origin=(float(origin[0]), float(origin[1]), float(plane_z)),
        grid_normal=(0.0, 0.0, 1.0),
        grid_axis1=(1.0, 0.0, 0.0),
        grid_alignment="bottom-left",
        grid_size=float(bounds.opts.length1),
        grid_size_extra=float(bounds.opts.length2),
        grid_spacing=spacing,
        grid_spacing_extra=spacing,
        n_length=length,
        n_radius=radius,
        n_opacity=opacity,
        n_sides=8,
        n_is_scalar_bar=False,
        n_is_reset_camera=False,
        nb_opacity=opacity,
        nb_radius=radius,
        nb_is_reset_camera=False,
        nd_opacity=nd_opacity,
        nd_radius=nd_radius,
        nd_sides=10,
        nd_is_reset_camera=False,
    )


def main() -> None:
    args = parse_args()
    q5 = load_q(args.data_dir, args.step)
    shape = q5.shape[:3]
    center = np.asarray(args.center, dtype=float)

    q_obj = n3d.QFieldObject(
        inputValue=InputQ(Q=q5, box_periodic_flag=PERIODIC_BOUNDARY),
        name=f"Q step {args.step}",
        is_detect_defects=True,
        is_classify_lines=True,
    )

    line_items = []
    for line in q_obj.lines:
        coords, smooth = smooth_or_raw(line, density=float(args.line_smooth_density))
        if len(coords) >= int(args.min_line_points):
            line_items.append((line, coords, smooth))

    if args.focus_loop and line_items:
        _, focus_coords, _ = max(line_items, key=lambda item: len(item[1]))
        center = np.mean(focus_coords, axis=0)
        hx, hy, hz = (float(v) for v in args.local_half_widths)
        x_bounds = None if args.local_x_bounds is None else tuple(float(v) for v in args.local_x_bounds)
        y_bounds = None if args.local_y_bounds is None else tuple(float(v) for v in args.local_y_bounds)
        z_bounds = None if args.local_z_bounds is None else tuple(float(v) for v in args.local_z_bounds)
        bounds = make_bounds(center, shape, hx, hy, hz, x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)
    else:
        x_bounds = None if args.local_x_bounds is None else tuple(float(v) for v in args.local_x_bounds)
        y_bounds = None if args.local_y_bounds is None else tuple(float(v) for v in args.local_y_bounds)
        z_bounds = None if args.local_z_bounds is None else tuple(float(v) for v in args.local_z_bounds)
        bounds = make_bounds(center, shape, float(args.x_half_width), x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)

    fig = PlotFigure(
        is_off_screen=True,
        name=f"beta overview step {args.step}",
        opts=OptsFigure(
            size=tuple(args.figure_size),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=tuple(center),
        ),
    )
    try:
        fig.pl.enable_anti_aliasing("ssaa")
    except Exception:
        try:
            fig.pl.enable_anti_aliasing()
        except Exception:
            pass
    try:
        fig.pl.enable_depth_peeling()
    except Exception:
        pass

    add_channel_shell(
        fig,
        bounds,
        int(args.surface_stride),
        opacity=float(args.shell_opacity),
    )
    add_director_plane(
        fig,
        q_obj,
        bounds,
        center,
        plane_z=float(center[2] if args.director_plane_z is None else args.director_plane_z),
        spacing=float(args.director_spacing),
        length=float(args.director_length),
        radius=float(args.director_radius),
        opacity=float(args.director_opacity),
        highlight_defect_directors=bool(args.highlight_defect_directors),
        highlight_radius=float(args.highlight_director_radius),
        highlight_opacity=float(args.highlight_director_opacity),
    )

    plotted = 0
    plotted_real_beta = 0
    for line, coords, smooth in line_items:
        mask = in_bounds(coords, bounds)
        if not np.any(mask):
            continue
        beta = (
            real_beta_for_line(line, coords, smooth, int(args.beta_samples))
            if args.real_beta
            else None
        )
        is_real_beta = beta is not None
        if not is_real_beta:
            beta = beta_proxy_from_tangent(coords)
        else:
            plotted_real_beta += 1
        plot_line(
            fig,
            coords,
            beta,
            radius=float(args.line_radius),
            sides=int(args.line_sides),
            colormap=str(args.line_colormap),
            opacity=float(args.line_opacity),
            ambient=float(args.line_ambient),
            diffuse=float(args.line_diffuse),
            specular=float(args.line_specular),
            colorbar_position=tuple(float(v) for v in args.colorbar_position),
            colorbar_size=tuple(float(v) for v in args.colorbar_size),
            show_scalar_bar=(plotted == 0 and not args.hide_colorbar),
        )
        plotted += 1

    if not args.hide_bounds:
        bounds.act_visualize(
            figure=fig,
            opts=n3d.OptsTube(
                radius=float(args.bounds_radius),
                color=(0.72, 0.72, 0.72),
                opacity=float(args.bounds_opacity),
                sides=8,
                is_scalar_bar=False,
                is_reset_camera=False,
            ),
        )

    if args.show_axes:
        axes_actor = fig.pl.add_axes(
            line_width=float(args.axes_line_width),
            x_color=(0.85, 0.10, 0.10),
            y_color=(0.10, 0.30, 0.85),
            z_color=(0.10, 0.75, 0.10),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            viewport=tuple(float(v) for v in args.axes_viewport),
        )
        axes_actor.SetXAxisLabelText("x")
        axes_actor.SetYAxisLabelText("y")
        axes_actor.SetZAxisLabelText("z")
        axes_colors = (
            (0.85, 0.10, 0.10),
            (0.02, 0.20, 0.55),
            (0.10, 0.75, 0.10),
        )
        caption_getters = (
            axes_actor.GetXAxisCaptionActor2D,
            axes_actor.GetYAxisCaptionActor2D,
            axes_actor.GetZAxisCaptionActor2D,
        )
        for getter, color in zip(caption_getters, axes_colors):
            caption = getter()
            caption.SetWidth(float(args.axes_label_box[0]))
            caption.SetHeight(float(args.axes_label_box[1]))
            text_prop = caption.GetCaptionTextProperty()
            text_prop.SetColor(*color)
            text_prop.SetItalic(True)
            text_prop.SetBold(True)
            text_prop.SetFontSize(int(args.axes_label_font_size))
        axes_actor.SetShaftTypeToLine()
        axes_actor.SetTipTypeToCone()
        axes_actor.SetNormalizedShaftLength(0.78, 0.78, 0.78)
        axes_actor.SetNormalizedTipLength(0.22, 0.22, 0.22)

    fig.act_view_isometric()
    fig.act_commit(
        focal_point=tuple(center),
        distance=float(args.camera_distance),
        azimuth=float(args.camera_azimuth),
        elevation=float(args.camera_elevation),
        roll=float(args.camera_roll),
    )
    fig.pl.camera.SetWindowCenter(
        float(args.camera_window_center[0]),
        float(args.camera_window_center[1]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.act_savefig(str(args.output), scale=float(args.save_scale))
    print(f"output={args.output}")
    print(f"lines_plotted={plotted}")
    if args.real_beta and plotted_real_beta == plotted and plotted > 0:
        print("beta_mode=real")
    elif args.real_beta and plotted_real_beta > 0:
        print(f"beta_mode=mixed real_lines={plotted_real_beta} proxy_lines={plotted - plotted_real_beta}")
    else:
        print("beta_mode=tangent_proxy")


if __name__ == "__main__":
    main()
