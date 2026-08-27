#!/usr/bin/env python3
"""Visualize how an optimized DAL activity protocol transports a loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pssolver.models.active_nematics import Q_convention_metadata, S_from_Q
from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.control import FunctionalSemiImplicitStep


SAMPLE_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
SLICE_TIMES = (0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPO_ROOT / "data_dal_smooth_trajectory_final",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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


def _optional_positive_int(parent: dict, key: str, path: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be null or a positive integer")
    return value


def _required_triple(
    parent: dict,
    key: str,
    path: str,
    *,
    integers: bool = False,
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
    if any(not np.isfinite(value) or value <= 0.0 for value in converted):
        raise ValueError(f"{path}.{key} must contain positive finite values")
    return converted


def load_run_config(result_dir: Path) -> dict:
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
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
    pressure_solver = _required_mapping(
        _required_mapping(metadata, "numerics", "metadata"),
        "pressure_solver",
        "metadata.numerics",
    )
    arguments = _required_mapping(
        _required_mapping(metadata, "optimization", "metadata"),
        "arguments",
        "metadata.optimization",
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
    core_threshold_fraction = _required_number(
        arguments,
        "core_threshold_fraction",
        "metadata.optimization.arguments",
        positive=True,
    )
    if core_threshold_fraction >= 1.0:
        raise ValueError(
            "metadata.optimization.arguments.core_threshold_fraction "
            "must be less than one"
        )

    return {
        "shape": _required_triple(
            solver,
            "shape",
            "metadata.solver",
            integers=True,
        ),
        "lengths": _required_triple(solver, "lengths", "metadata.solver"),
        "dt": _required_number(solver, "dt", "metadata.solver", positive=True),
        "steps": _required_positive_int(solver, "steps", "metadata.solver"),
        "S_bulk": _required_number(
            parameters,
            "S_bulk",
            "metadata.model.parameters",
            positive=True,
        ),
        "pressure_rel_tol": _required_number(
            pressure_solver,
            "relative_tolerance",
            "metadata.numerics.pressure_solver",
            positive=True,
        ),
        "pressure_max_iter": _required_positive_int(
            pressure_solver,
            "maximum_iterations",
            "metadata.numerics.pressure_solver",
        ),
        "pressure_fixed_iterations": _optional_positive_int(
            pressure_solver,
            "fixed_iterations",
            "metadata.numerics.pressure_solver",
        ),
        "block_size": _required_positive_int(
            arguments,
            "block_size",
            "metadata.optimization.arguments",
        ),
        "alpha_limits": (alpha_min, alpha_max),
        "initial_alpha": _required_number(
            arguments,
            "initial_alpha",
            "metadata.optimization.arguments",
        ),
        "core_threshold_fraction": core_threshold_fraction,
        "core_transition_fraction": _required_number(
            arguments,
            "core_transition_fraction",
            "metadata.optimization.arguments",
            positive=True,
        ),
    }


def q5_to_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(values).movedim(-1, 0).unsqueeze(1).to(device)


def disturbance_density(q: torch.Tensor, S_bulk: float) -> torch.Tensor:
    uniform = torch.zeros_like(q)
    uniform[0] = S_bulk
    uniform[3] = -S_bulk / 2.0
    return (q - uniform).square().sum(dim=0)


def disturbance_center_x(
    q: torch.Tensor,
    spatial_mask: torch.Tensor,
    S_bulk: float,
    period_x: float,
) -> float:
    density = disturbance_density(q, S_bulk) * spatial_mask.unsqueeze(0)
    weight_x = density.sum(dim=(2, 3), dtype=torch.float64)[0]
    nx = q.shape[2]
    x = (
        torch.arange(nx, device=q.device, dtype=q.dtype) + 0.5
    ) * period_x / nx
    theta = 2.0 * torch.pi * x / period_x
    center = (
        torch.remainder(
            torch.atan2(
                (weight_x * torch.sin(theta)).sum(),
                (weight_x * torch.cos(theta)).sum(),
            ),
            2.0 * torch.pi,
        )
        * period_x
        / (2.0 * torch.pi)
    )
    return float(center.item())


def core_center_x(
    q: torch.Tensor,
    spatial_mask: torch.Tensor,
    S_bulk: float,
    threshold_fraction: float,
    transition_fraction: float,
    period_x: float,
) -> float:
    qxx, qxy, qxz, qyy, qyz = q
    tr_q_squared = (
        qxx.square()
        + qyy.square()
        + (qxx + qyy).square()
        + 2.0 * (qxy.square() + qxz.square() + qyz.square())
    )
    magnitude = torch.sqrt(
        (2.0 / 3.0) * tr_q_squared.clamp_min(0.0) + torch.finfo(q.dtype).eps
    )
    threshold = threshold_fraction * S_bulk
    transition = transition_fraction * S_bulk
    density = torch.sigmoid((threshold - magnitude) / transition)
    bulk_density = torch.sigmoid(
        torch.as_tensor(
            (threshold - S_bulk) / transition,
            device=q.device,
            dtype=q.dtype,
        )
    )
    density = ((density - bulk_density) / (1.0 - bulk_density)).square()
    density = density * spatial_mask.unsqueeze(0)
    weight_x = density.sum(dim=(2, 3), dtype=torch.float64)[0]
    nx = q.shape[2]
    x = (
        torch.arange(nx, device=q.device, dtype=q.dtype) + 0.5
    ) * period_x / nx
    theta = 2.0 * torch.pi * x / period_x
    center = (
        torch.remainder(
            torch.atan2(
                (weight_x * torch.sin(theta)).sum(),
                (weight_x * torch.cos(theta)).sum(),
            ),
            2.0 * torch.pi,
        )
        * period_x
        / (2.0 * torch.pi)
    )
    return float(center.item())


def periodic_delta(value: float, reference: float, period: float) -> float:
    return (value - reference + 0.5 * period) % period - 0.5 * period


def activity_fields(amplitudes: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.einsum("ti,ixyz->txyz", amplitudes, masks, optimize=True)


def effective_activity(activity: np.ndarray, masks: np.ndarray) -> np.ndarray:
    envelope = masks.sum(axis=0)
    denominator = envelope.sum(axis=(1, 2))
    numerator = activity.sum(axis=(2, 3))
    return np.divide(
        numerator,
        denominator[None, :],
        out=np.full_like(numerator, np.nan),
        where=denominator[None, :] > 1e-10,
    )


def block_for_time(time: float, total_time: float, num_blocks: int) -> int:
    if time >= total_time:
        return num_blocks - 1
    return min(int(time / total_time * num_blocks), num_blocks - 1)




def replay(
    result_dir: Path,
    config: dict,
    masks: np.ndarray,
    amplitudes: np.ndarray,
    sample_stride: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    cache_path = result_dir / "activity_trajectory_cache.npz"
    if cache_path.is_file():
        cache = np.load(cache_path)
        cached_stride = int(cache["sample_stride"])
        if cached_stride == sample_stride and "core_centers_x" in cache.files:
            selected = {
                float(key.removeprefix("q_t")): cache[key]
                for key in cache.files
                if key.startswith("q_t")
            }
            return (
                cache["times"],
                cache["centers_x"],
                cache["core_centers_x"],
                selected,
            )

    initial_values = np.load(result_dir / "Q_initial.npy")
    target_values = np.load(result_dir / "Q_target_downsampled.npy")
    initial_q = q5_to_tensor(initial_values, device)
    target_q = q5_to_tensor(target_values, device)
    spatial_mask = torch.from_numpy(
        np.load(result_dir / "control_mask.npy")
    ).to(device=device, dtype=initial_q.dtype)
    initial_fields = {
        name: initial_q[index, 0] for index, name in enumerate(Q_COMPONENTS)
    }
    solver = build_active_nematic_channel(
        config["shape"],
        config["lengths"],
        config["dt"],
        initial_fields,
        device=device,
        batchsize=1,
        pressure_rel_tol=config["pressure_rel_tol"],
        pressure_max_iter=config["pressure_max_iter"],
        pressure_fixed_iterations=config["pressure_fixed_iterations"],
    )
    stepper = FunctionalSemiImplicitStep(solver)
    masks_t = torch.from_numpy(masks).to(device=device, dtype=initial_q.dtype)
    amplitudes_t = torch.from_numpy(amplitudes).to(device=device, dtype=initial_q.dtype)
    q = initial_q
    steps = config["steps"]
    block_size = config["block_size"]
    dt = config["dt"]
    selected_steps = {
        round(time * steps): time for time in SLICE_TIMES
    }
    times = []
    centers = []
    core_centers = []
    selected: dict[float, np.ndarray] = {}

    def record(step: int) -> None:
        times.append(step * dt)
        centers.append(
            disturbance_center_x(
                q,
                spatial_mask,
                config["S_bulk"],
                config["lengths"][0],
            )
        )
        core_centers.append(
            core_center_x(
                q,
                spatial_mask,
                config["S_bulk"],
                config["core_threshold_fraction"],
                config["core_transition_fraction"],
                config["lengths"][0],
            )
        )
        if step in selected_steps:
            selected[selected_steps[step]] = (
                q.detach().cpu().squeeze(1).movedim(0, -1).numpy()
            )

    with torch.no_grad():
        record(0)
        for step in range(steps):
            block = min(step // block_size, amplitudes.shape[0] - 1)
            alpha = torch.einsum("m,mxyz->xyz", amplitudes_t[block], masks_t)
            q = stepper(q, alpha.unsqueeze(0))
            next_step = step + 1
            if next_step % sample_stride == 0 or next_step == steps:
                record(next_step)

    payload = {
        "times": np.asarray(times),
        "centers_x": np.asarray(centers),
        "core_centers_x": np.asarray(core_centers),
        "sample_stride": np.asarray(sample_stride),
    }
    payload.update({f"q_t{time}": values for time, values in selected.items()})
    np.savez_compressed(cache_path, **payload)
    return payload["times"], payload["centers_x"], payload["core_centers_x"], selected


def reference_trajectory(
    times: np.ndarray,
    total_time: float,
    initial_center: float,
    target_center: float,
    period_x: float,
) -> np.ndarray:
    tau = np.clip(times / total_time, 0.0, 1.0)
    progress = tau**2 * (3.0 - 2.0 * tau)
    displacement = periodic_delta(target_center, initial_center, period_x)
    return initial_center + progress * displacement


def plot_spacetime(
    output: Path,
    alpha_xt: np.ndarray,
    baseline: float,
    x_edges: np.ndarray,
    time_edges: np.ndarray,
    trajectory_times: np.ndarray,
    actual_center: np.ndarray,
    core_center: np.ndarray,
    reference_center: np.ndarray,
    target_center: float,
    alpha_limits: tuple[float, float],
) -> None:
    delta = alpha_xt - baseline
    delta_limit = max(float(np.nanmax(np.abs(delta))), 1e-6)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), dpi=190, sharex=True, constrained_layout=True)
    panels = (
        (alpha_xt.T, "viridis", alpha_limits[0], alpha_limits[1], r"$\alpha_{\mathrm{eff}}$"),
        (delta.T, "RdBu_r", -delta_limit, delta_limit, r"$\Delta\alpha_{\mathrm{eff}}$"),
    )
    for axis, (data, cmap, vmin, vmax, label) in zip(axes, panels):
        image = axis.pcolormesh(
            time_edges, x_edges, data, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axis.plot(trajectory_times, actual_center, color="black", linewidth=2.2, label="Q-disturbance center")
        axis.plot(
            trajectory_times,
            core_center,
            color="#00D5D8",
            linewidth=1.8,
            linestyle="-.",
            label="Coarse core center",
        )
        axis.plot(trajectory_times, reference_center, color="white", linewidth=2.0, linestyle="--", label="Reference trajectory")
        axis.axhline(target_center, color="#E5C100", linewidth=1.8, linestyle=":", label="Target center")
        axis.set_ylim(56.0, 70.0)
        axis.set_ylabel("x")
        axis.legend(loc="upper right", framealpha=0.9, fontsize=9)
        fig.colorbar(image, ax=axis, pad=0.015, label=label)
    axes[0].set_title("Applied activity and loop trajectory")
    axes[1].set_title(r"Activity redistribution relative to $\alpha_0=4.5$")
    axes[1].set_xlabel("Time")
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_profiles(
    output: Path,
    alpha_xt: np.ndarray,
    baseline: float,
    x: np.ndarray,
    total_time: float,
    trajectory_times: np.ndarray,
    actual_center: np.ndarray,
    reference_center: np.ndarray,
    target_center: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=190, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(SAMPLE_TIMES)))
    for time, color in zip(SAMPLE_TIMES, colors):
        block = block_for_time(time, total_time, alpha_xt.shape[0])
        actual = float(np.interp(time, trajectory_times, actual_center))
        reference = float(np.interp(time, trajectory_times, reference_center))
        axes[0].plot(x, alpha_xt[block], color=color, linewidth=2.2, label=f"t={time:g}")
        axes[1].plot(x, alpha_xt[block] - baseline, color=color, linewidth=2.2, label=f"t={time:g}")
        axes[0].scatter(actual, np.interp(actual, x, alpha_xt[block]), color=color, edgecolor="black", s=42, zorder=4)
        axes[1].scatter(actual, np.interp(actual, x, alpha_xt[block] - baseline), color=color, edgecolor="black", s=42, zorder=4)
        for axis in axes:
            axis.axvline(reference, color=color, linewidth=0.9, linestyle="--", alpha=0.7)
    for axis in axes:
        axis.axvline(target_center, color="#C5A800", linewidth=1.5, linestyle=":")
        axis.set_xlim(56.0, 70.0)
        axis.set_xlabel("x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\alpha_{\mathrm{eff}}$")
    axes[1].set_ylabel(r"$\Delta\alpha_{\mathrm{eff}}$")
    axes[0].set_title("Absolute applied activity")
    axes[1].set_title(r"Redistribution relative to $\alpha_0=4.5$")
    axes[0].legend(frameon=False, ncols=3)
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="black", label="Q-disturbance center"),
            Line2D([0], [0], color="#777777", linestyle="--", label="Reference center"),
            Line2D([0], [0], color="#C5A800", linestyle=":", label="Target center"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_spatial_slices(
    output: Path,
    activity: np.ndarray,
    selected_q: dict[float, np.ndarray],
    config: dict,
    trajectory_times: np.ndarray,
    actual_center: np.ndarray,
    core_center: np.ndarray,
    reference_center: np.ndarray,
) -> None:
    shape = activity.shape[1:]
    spacing = tuple(
        length / size for length, size in zip(config["lengths"], shape)
    )
    x_edges = np.arange(shape[0] + 1) * spacing[0]
    y_edges = np.arange(shape[1] + 1) * spacing[1]
    z_index = int(np.argmin(np.abs((np.arange(shape[2]) + 0.5) * spacing[2] - 5.0)))
    total_time = config["steps"] * config["dt"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=190, sharey=True, constrained_layout=True)
    image = None
    for axis, time in zip(axes, SLICE_TIMES):
        block = block_for_time(time, total_time, activity.shape[0])
        image = axis.pcolormesh(
            x_edges,
            y_edges,
            activity[block, :, :, z_index].T,
            shading="flat",
            cmap="viridis",
            vmin=config["alpha_limits"][0],
            vmax=config["alpha_limits"][1],
        )
        actual = float(np.interp(time, trajectory_times, actual_center))
        core = float(np.interp(time, trajectory_times, core_center))
        reference = float(np.interp(time, trajectory_times, reference_center))
        axis.axvline(actual, color="black", linewidth=2.0, label="Q-disturbance center")
        axis.axvline(
            core,
            color="#00D5D8",
            linewidth=1.8,
            linestyle="-.",
            label="Coarse core center",
        )
        axis.axvline(reference, color="white", linewidth=1.8, linestyle="--", label="Reference")
        q5 = selected_q.get(time)
        if q5 is not None:
            deficit = np.maximum(
                (config["S_bulk"] - np.asarray(S_from_Q(q5)))
                / config["S_bulk"],
                0.0,
            )
            x_centers = (np.arange(shape[0]) + 0.5) * spacing[0]
            y_centers = (np.arange(shape[1]) + 0.5) * spacing[1]
            axis.contour(
                x_centers,
                y_centers,
                deficit[:, :, z_index].T,
                levels=[0.2],
                colors=["#F5F5F5"],
                linewidths=1.4,
            )
        axis.set_xlim(56.0, 70.0)
        axis.set_ylim(1.0, 9.0)
        axis.set_xlabel("x")
        axis.set_title(f"t={time:g}, z≈5")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(image, ax=axes, pad=0.015, label=r"$\alpha(x,y,z=5,t)$")
    fig.suptitle("Spatial localization of the applied activity", fontsize=16)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_mask_coefficients(
    output: Path,
    amplitudes: np.ndarray,
    total_time: float,
    alpha_limits: tuple[float, float],
) -> None:
    time_edges = np.linspace(0.0, total_time, amplitudes.shape[0] + 1)
    mask_edges = np.arange(amplitudes.shape[1] + 1) - 0.5
    fig, axis = plt.subplots(figsize=(10.5, 4.8), dpi=190, constrained_layout=True)
    image = axis.pcolormesh(
        time_edges,
        mask_edges,
        amplitudes.T,
        shading="flat",
        cmap="viridis",
        vmin=alpha_limits[0],
        vmax=alpha_limits[1],
    )
    axis.set_yticks(np.arange(amplitudes.shape[1]))
    axis.set_xlabel("Time")
    axis.set_ylabel("Path-Gaussian mask index")
    axis.set_title(r"Optimized control parameters $a_i(t)$")
    fig.colorbar(image, ax=axis, pad=0.015, label=r"$a_i$")
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (
        args.output_dir or result_dir / "activity_visualization"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_run_config(result_dir)
    shape = config["shape"]
    S_bulk = config["S_bulk"]
    masks = np.load(result_dir / "control_masks.npy")
    if masks.ndim != 4 or masks.shape[1:] != shape:
        raise ValueError(
            f"control_masks.npy has shape {masks.shape}; expected (M, {shape})"
        )
    amplitudes = np.load(result_dir / "alpha_amplitudes.npy")
    activity = activity_fields(amplitudes, masks)
    alpha_xt = effective_activity(activity, masks)
    total_time = config["steps"] * config["dt"]
    spacing = tuple(
        length / size for length, size in zip(config["lengths"], shape)
    )
    x = (np.arange(shape[0]) + 0.5) * spacing[0]
    x_edges = np.arange(shape[0] + 1) * spacing[0]
    time_edges = np.linspace(0.0, total_time, amplitudes.shape[0] + 1)
    alpha_limits = config["alpha_limits"]
    baseline = config["initial_alpha"]

    trajectory_times, actual_center, core_center, selected_q = replay(
        result_dir,
        config,
        masks,
        amplitudes,
        args.sample_stride,
        torch.device(args.device),
    )
    initial_values = np.load(result_dir / "Q_initial.npy")
    target_values = np.load(result_dir / "Q_target_downsampled.npy")
    for name, values in (("initial", initial_values), ("target", target_values)):
        if values.shape != shape + (5,):
            raise ValueError(
                f"{name} Q has shape {values.shape}; expected {shape + (5,)}"
            )
    device = torch.device(args.device)
    initial_q = q5_to_tensor(initial_values, device)
    target_q = q5_to_tensor(target_values, device)
    spatial_mask = torch.from_numpy(np.load(result_dir / "control_mask.npy")).to(
        device=args.device, dtype=initial_q.dtype
    )
    initial_center = disturbance_center_x(
        initial_q,
        spatial_mask,
        S_bulk,
        config["lengths"][0],
    )
    target_center = disturbance_center_x(
        target_q,
        spatial_mask,
        S_bulk,
        config["lengths"][0],
    )
    reference_center = reference_trajectory(
        trajectory_times,
        total_time,
        initial_center,
        target_center,
        config["lengths"][0],
    )

    plot_spacetime(
        output_dir / "activity_spacetime_and_trajectory.png",
        alpha_xt,
        baseline,
        x_edges,
        time_edges,
        trajectory_times,
        actual_center,
        core_center,
        reference_center,
        target_center,
        alpha_limits,
    )
    plot_profiles(
        output_dir / "activity_profiles.png",
        alpha_xt,
        baseline,
        x,
        total_time,
        trajectory_times,
        actual_center,
        reference_center,
        target_center,
    )
    plot_spatial_slices(
        output_dir / "activity_spatial_slices.png",
        activity,
        selected_q,
        config,
        trajectory_times,
        actual_center,
        core_center,
        reference_center,
    )
    plot_mask_coefficients(
        output_dir / "mask_amplitudes.png",
        amplitudes,
        total_time,
        alpha_limits,
    )
    np.savez(
        output_dir / "activity_trajectory.npz",
        trajectory_times=trajectory_times,
        actual_center_x=actual_center,
        coarse_core_center_x=core_center,
        reference_center_x=reference_center,
        target_center_x=target_center,
        alpha_effective_xt=alpha_xt,
        delta_alpha_effective_xt=alpha_xt - baseline,
    )
    summary = {
        "initial_alpha": baseline,
        "initial_center_x": initial_center,
        "initial_coarse_core_center_x": float(core_center[0]),
        "target_center_x": target_center,
        "final_actual_center_x": float(actual_center[-1]),
        "final_coarse_core_center_x": float(core_center[-1]),
        "final_reference_center_x": float(reference_center[-1]),
        "activity_min": float(np.nanmin(alpha_xt)),
        "activity_max": float(np.nanmax(alpha_xt)),
        "sample_stride": args.sample_stride,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
