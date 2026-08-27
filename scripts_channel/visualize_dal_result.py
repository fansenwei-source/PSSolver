#!/usr/bin/env python3
"""Create a report-ready visualization of one Channel DAL result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for candidate in (REPO_ROOT.parent / "Nematics3D" / "src", REPO_ROOT / "Nematics3D" / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q

from pssolver.models.active_nematics import Q_convention_metadata


DEFAULT_RESULT = REPO_ROOT / "data_dal_pure_splay_x64_to_x62_T1_coreaware"
COLORS = {
    "Initial": "#2474B5",
    "Optimized": "#D9781F",
    "Target": "#159570",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    return parser.parse_args()


def _required_mapping(parent: dict, key: str, path: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _required_number(
    parent: dict,
    key: str,
    path: str,
    *,
    positive: bool = False,
) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be numeric")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{path}.{key} must be {qualifier}")
    return number


def _required_positive_int(parent: dict, key: str, path: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")
    return value


def _required_triple(
    parent: dict,
    key: str,
    path: str,
    *,
    integers: bool = False,
    positive: bool = False,
) -> tuple[int, int, int] | tuple[float, float, float]:
    values = parent.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{path}.{key} must contain three values")
    if integers:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError(f"{path}.{key} must contain positive integers")
        return tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{path}.{key} must contain numeric values")
    converted = tuple(float(value) for value in values)
    if any(
        not np.isfinite(value) or (positive and value <= 0.0)
        for value in converted
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{path}.{key} must contain {qualifier} values")
    return converted


def load_run_config(result_dir: Path) -> dict:
    with (result_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    if metadata.get("schema_version") != 1:
        raise ValueError("metadata must declare schema_version=1")
    solver = _required_mapping(metadata, "solver", "metadata")
    model = _required_mapping(metadata, "model", "metadata")
    if model.get("name") != "active_nematics":
        raise ValueError("metadata.model.name must be 'active_nematics'")
    convention = model.get("Q_convention")
    expected_convention = Q_convention_metadata()
    if convention != expected_convention:
        raise ValueError(
            "metadata.model.Q_convention must equal the complete canonical "
            f"convention {expected_convention!r}"
        )
    parameters = _required_mapping(model, "parameters", "metadata.model")
    optimization = _required_mapping(metadata, "optimization", "metadata")
    arguments = _required_mapping(
        optimization,
        "arguments",
        "metadata.optimization",
    )
    initial_input = _required_mapping(
        _required_mapping(metadata, "initial_condition", "metadata"),
        "input",
        "metadata.initial_condition",
    )
    target_input = _required_mapping(
        _required_mapping(metadata, "target", "metadata"),
        "input",
        "metadata.target",
    )

    alpha_min = _required_number(
        arguments,
        "alpha_min",
        "metadata.optimization.arguments",
    )
    alpha_max = _required_number(
        arguments,
        "alpha_max",
        "metadata.optimization.arguments",
    )
    if alpha_min >= alpha_max:
        raise ValueError(
            "metadata.optimization.arguments.alpha_min must be less than alpha_max"
        )
    loss_function = arguments.get("loss_function")
    if not isinstance(loss_function, str) or not loss_function:
        raise ValueError(
            "metadata.optimization.arguments.loss_function must be a non-empty string"
        )

    return {
        "shape": _required_triple(
            solver,
            "shape",
            "metadata.solver",
            integers=True,
        ),
        "lengths": _required_triple(
            solver,
            "lengths",
            "metadata.solver",
            positive=True,
        ),
        "dt": _required_number(solver, "dt", "metadata.solver", positive=True),
        "steps": _required_positive_int(solver, "steps", "metadata.solver"),
        "S_bulk": _required_number(
            parameters,
            "S_bulk",
            "metadata.model.parameters",
            positive=True,
        ),
        "initial_loop": (
            _required_triple(initial_input, "center", "metadata.initial_condition.input"),
            _required_number(
                initial_input,
                "radius",
                "metadata.initial_condition.input",
                positive=True,
            ),
        ),
        "target_loop": (
            _required_triple(target_input, "center", "metadata.target.input"),
            _required_number(
                target_input,
                "radius",
                "metadata.target.input",
                positive=True,
            ),
        ),
        "alpha_limits": (alpha_min, alpha_max),
        "loss_function": loss_function,
        "running_comoving_q_weight": _required_number(
            arguments,
            "running_comoving_q_weight",
            "metadata.optimization.arguments",
        ),
        "terminal_comoving_q_weight": _required_number(
            arguments,
            "terminal_comoving_q_weight",
            "metadata.optimization.arguments",
        ),
    }


def disturbance_profile(q5: np.ndarray, S_bulk: float) -> np.ndarray:
    background = np.array(
        [S_bulk, 0.0, 0.0, -S_bulk / 2.0, 0.0],
        dtype=q5.dtype,
    )
    profile = np.sum((q5 - background) ** 2, axis=(1, 2, 3))
    maximum = float(np.max(profile))
    return profile / maximum if maximum > 0 else profile


def detected_loop_tube(q5: np.ndarray, spacing: tuple[float, float, float]) -> pv.PolyData:
    director = director_from_Q(q5)
    defects = n3d.defect_detect(
        director,
        threshold=0.0,
        is_boundary_periodic=(True, False, False),
        planes=(True, True, True),
    )
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=(q5.shape[0], np.inf, np.inf),
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    loops = [line for line in lines if getattr(line, "kind", None) == "loop"]
    if not loops:
        return pv.PolyData()
    line = max(loops, key=lambda item: int(getattr(item, "calc_defect_num", 0)))
    coords = np.asarray(line.calc_defect_coords, dtype=float)
    if coords.shape[0] >= 5:
        try:
            coords = np.asarray(
                line.act_smooth(window_length=5, min_line_length=5, is_window_warning=False).result,
                dtype=float,
            )
        except Exception:
            pass
    physical = coords * np.asarray(spacing) + 0.5 * np.asarray(spacing)
    return pv.lines_from_points(physical, close=True).tube(radius=0.11, n_sides=24)


def analytic_loop_tube(center: tuple[float, float, float], radius: float) -> pv.PolyData:
    angle = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    points = np.column_stack(
        (
            np.full_like(angle, center[0]),
            center[1] + radius * np.cos(angle),
            center[2] + radius * np.sin(angle),
        )
    )
    return pv.lines_from_points(points, close=True).tube(radius=0.11, n_sides=24)


def render_loops(
    q_fields: dict[str, np.ndarray],
    reference_loops: dict[str, tuple[tuple[float, float, float], float]],
    spacing: tuple[float, float, float],
    window: tuple[float, float, float, float, float, float],
    output: Path,
) -> None:
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(shape=(1, 4), off_screen=True, window_size=(2160, 620), border=False)
    plotter.set_background("white")
    loop_tubes = {
        name: (
            analytic_loop_tube(*reference_loops[name])
            if name in reference_loops
            else detected_loop_tube(q_fields[name], spacing)
        )
        for name in q_fields
    }
    panels = [(name, [name]) for name in q_fields] + [("Overlay", list(q_fields))]
    for index, (title, names) in enumerate(panels):
        plotter.subplot(0, index)
        for name in names:
            plotter.add_mesh(
                loop_tubes[name],
                color=COLORS[name],
                opacity=1.0 if len(names) == 1 else 0.9,
                smooth_shading=True,
                specular=0.35,
            )
        plotter.add_title(title, font_size=17, color="#20252A")
        plotter.show_bounds(
            bounds=window,
            axes_ranges=window,
            xtitle="x",
            ytitle="y",
            ztitle="z",
            font_size=11,
            color="#4B535A",
            grid="back",
            location="outer",
            all_edges=True,
        )
        plotter.camera_position = [
            (window[1] + 8.0, window[3] + 10.0, window[5] + 8.0),
            ((window[0] + window[1]) / 2, (window[2] + window[3]) / 2, (window[4] + window[5]) / 2),
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.zoom(1.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output), scale=2)
    plotter.close()


def effective_activity_xt(
    amplitudes: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Reconstruct alpha(x,y,z,t) and average it over the controlled cross-section."""

    if amplitudes.ndim != 2 or masks.ndim != 4:
        raise ValueError("Expected amplitudes (Nt, Nm) and masks (Nm, Nx, Ny, Nz).")
    if amplitudes.shape[1] != masks.shape[0]:
        raise ValueError("The number of control amplitudes must match the masks.")

    alpha = np.einsum("ti,ixyz->txyz", amplitudes, masks, optimize=True)
    control_envelope = masks.sum(axis=0)
    cross_section_weight = control_envelope.sum(axis=(1, 2))
    alpha_integral = alpha.sum(axis=(2, 3))
    return np.divide(
        alpha_integral,
        cross_section_weight[None, :],
        out=np.full_like(alpha_integral, np.nan),
        where=cross_section_weight[None, :] > 1e-10,
    )


def plot_diagnostics(
    q_fields: dict[str, np.ndarray],
    amplitudes: np.ndarray,
    masks: np.ndarray,
    S_bulk: float,
    spacing: tuple[float, float, float],
    dt: float,
    steps: int,
    alpha_limits: tuple[float, float],
    output: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, (ax_profile, ax_control) = plt.subplots(1, 2, figsize=(13.5, 4.7), dpi=180, constrained_layout=True)

    x = (np.arange(next(iter(q_fields.values())).shape[0]) + 0.5) * spacing[0]
    for name, q5 in q_fields.items():
        ax_profile.plot(x, disturbance_profile(q5, S_bulk), color=COLORS[name], linewidth=2.5, label=name)
    ax_profile.set_xlim(56, 70)
    ax_profile.set_ylim(bottom=0)
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel("Normalized loop signature")
    ax_profile.set_title("Translation along x")
    ax_profile.grid(alpha=0.22)
    ax_profile.legend(frameon=False, ncols=3, loc="upper left")

    data = effective_activity_xt(amplitudes, masks).T
    total_time = steps * dt
    time_edges = np.linspace(0.0, total_time, amplitudes.shape[0] + 1)
    x_edges = np.arange(masks.shape[1] + 1, dtype=float) * spacing[0]
    image = ax_control.pcolormesh(
        time_edges,
        x_edges,
        data,
        cmap="RdYlBu_r",
        vmin=float(alpha_limits[0]),
        vmax=float(alpha_limits[1]),
        shading="flat",
    )
    ax_control.set_xlim(0.0, total_time)
    ax_control.set_ylim(56.0, 70.0)
    ax_control.set_xlabel("Time")
    ax_control.set_ylabel("x")
    ax_control.set_title(r"Applied activity $\alpha_{\mathrm{eff}}(x,t)$")
    colorbar = fig.colorbar(image, ax=ax_control, pad=0.02)
    colorbar.set_label(r"$\alpha_{\mathrm{eff}}$")
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def trim_white(image: Image.Image, threshold: int = 248) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    mask = np.any(array < threshold, axis=2)
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
    return image.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), min(image.width, xs.max() + 9), min(image.height, ys.max() + 9)))


def compose_summary(
    core_path: Path,
    diagnostic_path: Path,
    output: Path,
    loss_function: str,
) -> None:
    core = trim_white(Image.open(core_path).convert("RGB"))
    diagnostic = trim_white(Image.open(diagnostic_path).convert("RGB"))
    width = 2100
    core.thumbnail((width, 670), Image.Resampling.LANCZOS)
    diagnostic.thumbnail((width - 80, 720), Image.Resampling.LANCZOS)
    title_height = 132
    gap = 24
    canvas = Image.new("RGB", (width, title_height + core.height + gap + diagnostic.height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = None
        subtitle_font = None
    loss_titles = {
        "core_aware_comoving_q_trajectory": "Comoving-Q trajectory",
        "core_aware_q_tracking": "Core-aware Q tracking",
        "core_translation": "Translation with core-mass preservation",
        "core_translation_no_mass": "Translation only (no core-mass penalty)",
        "smooth_trajectory": "Smooth trajectory",
    }
    loss_title = loss_titles.get(loss_function, loss_function.replace("_", " ").title())
    draw.text(
        (48, 28),
        f"DAL: {loss_title}",
        fill="#20252A",
        font=font,
    )
    draw.text(
        (50, 84),
        "Reference loops (Initial/Target) and coarse detected loop (Optimized)",
        fill="#626A70",
        font=subtitle_font,
    )
    canvas.paste(core, ((width - core.width) // 2, title_height))
    canvas.paste(diagnostic, ((width - diagnostic.width) // 2, title_height + core.height + gap))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir / "visualization").resolve()
    config = load_run_config(result_dir)
    shape = config["shape"]
    spacing = tuple(
        length / size for length, size in zip(config["lengths"], shape)
    )
    S_bulk = config["S_bulk"]
    q_fields = {
        "Initial": np.load(result_dir / "Q_initial.npy"),
        "Optimized": np.load(result_dir / "Q_optimized_final.npy"),
        "Target": np.load(result_dir / "Q_target_downsampled.npy"),
    }
    for name, q5 in q_fields.items():
        if q5.shape != shape + (5,):
            raise ValueError(f"{name} Q shape is {q5.shape}, expected {shape + (5,)}")
    reference_loops = {
        "Initial": config["initial_loop"],
        "Target": config["target_loop"],
    }
    window = tuple(args.window) if args.window else (56.0, 70.0, 1.5, 8.5, 1.5, 8.5)

    core_path = output_dir / "loop_comparison.png"
    diagnostic_path = output_dir / "translation_and_control.png"
    summary_path = output_dir / "dal_result_summary.png"
    render_loops(
        q_fields,
        reference_loops,
        spacing,
        window,
        core_path,
    )
    plot_diagnostics(
        q_fields,
        np.load(result_dir / "alpha_amplitudes.npy"),
        np.load(result_dir / "control_masks.npy"),
        S_bulk,
        spacing,
        config["dt"],
        config["steps"],
        config["alpha_limits"],
        diagnostic_path,
    )
    loss_function = config["loss_function"]
    if (
        loss_function == "core_aware_comoving_q_trajectory"
        and config["running_comoving_q_weight"] == 0.0
        and config["terminal_comoving_q_weight"] == 0.0
    ):
        loss_function = "smooth_trajectory"
    compose_summary(
        core_path,
        diagnostic_path,
        summary_path,
        loss_function,
    )
    print(f"loop_comparison={core_path}")
    print(f"diagnostics={diagnostic_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
