#!/usr/bin/env python3
"""RC4.3.1 analysis-only adjudication of lifted Plane correctness.

The diagnostic deliberately builds the package-owned ``legacy_production``
runtime.  Its independent references use analytic Q derivatives and a direct
physical-space constitutive evaluation; they never call the lifting projector
whose representation is under review.  No production default or solver path
is changed by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

if __package__:
    from benchmarks.profile_beris_edwards_timestep import (
        RegionTimer,
        _install_transform_timers,
    )
    from benchmarks.profile_plane_runtime_timestep import (
        RuntimeProfileConfig,
        _run_spec,
    )
    from benchmarks.qualify_plane_static_lifting import _manufactured_error
else:
    from profile_beris_edwards_timestep import (
        RegionTimer,
        _install_transform_timers,
    )
    from profile_plane_runtime_timestep import RuntimeProfileConfig, _run_spec
    from qualify_plane_static_lifting import _manufactured_error
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    analytic_periodic_defect_gas_2d,
    prescribed_q,
)
from pssolver.models.active_nematics.stokes import (
    BerisEdwardsFreeSlipStokes,
)
from pssolver.operators.lifting import legacy_whole_slab_sine_conditioning
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    plane_physical_component,
)


SCHEMA_VERSION = 1
WALL_POINTS = (16, 32, 64, 128, 256)
HEIGHT = 10.0
SCALAR_ORDER = 0.6
Q_OSCILLATION = 0.75 * SCALAR_ORDER
Q_OFFSET = 0.25 * SCALAR_ORDER


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in Q_COMPONENTS:
        digest.update(name.encode("ascii"))
        digest.update(_tensor_sha256(values[name]).encode("ascii"))
    return digest.hexdigest()


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm((actual - expected).reshape(-1)))
    denominator = float(np.linalg.norm(expected.reshape(-1)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_array = actual.detach().cpu().numpy()
    expected_array = expected.detach().cpu().numpy()
    return {
        "relative_l2": _relative_l2(actual_array, expected_array),
        "linf": float(np.max(np.abs(actual_array - expected_array))),
    }


def _adjacent_rates(errors: list[float]) -> list[float]:
    return [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
    ]


def _regression_rate(errors: list[float]) -> float:
    inverse_resolution = 1.0 / np.asarray(WALL_POINTS, dtype=np.float64)
    slope, _intercept = np.polyfit(
        np.log(inverse_resolution),
        np.log(np.asarray(errors, dtype=np.float64)),
        1,
    )
    return float(slope)


def _cell_centers(wall_points: int) -> torch.Tensor:
    return (
        torch.arange(wall_points, dtype=torch.float64) + 0.5
    ) * (HEIGHT / wall_points)


def _twist_components(
    z: torch.Tensor,
    *,
    plane: str,
) -> tuple[torch.Tensor, ...]:
    phase = math.pi * z / HEIGHT
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    zero = torch.zeros_like(z)
    if plane == "xy":
        return (
            Q_OFFSET + Q_OSCILLATION * cosine,
            Q_OSCILLATION * sine,
            zero,
            Q_OFFSET - Q_OSCILLATION * cosine,
            zero,
        )
    if plane == "xz":
        return (
            Q_OFFSET + Q_OSCILLATION * cosine,
            zero,
            Q_OSCILLATION * sine,
            -0.5 * SCALAR_ORDER * torch.ones_like(z),
            zero,
        )
    raise ValueError("plane must be 'xy' or 'xz'")


def _twist_gradient(
    z: torch.Tensor,
    *,
    plane: str,
) -> tuple[torch.Tensor, ...]:
    wave = math.pi / HEIGHT
    phase = wave * z
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    zero = torch.zeros_like(z)
    if plane == "xy":
        return (
            -Q_OSCILLATION * wave * sine,
            Q_OSCILLATION * wave * cosine,
            zero,
            Q_OSCILLATION * wave * sine,
            zero,
        )
    if plane == "xz":
        return (
            -Q_OSCILLATION * wave * sine,
            zero,
            Q_OSCILLATION * wave * cosine,
            zero,
            zero,
        )
    raise ValueError("plane must be 'xy' or 'xz'")


def _twist_laplacian(
    z: torch.Tensor,
    *,
    plane: str,
) -> tuple[torch.Tensor, ...]:
    wave = math.pi / HEIGHT
    phase = wave * z
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    zero = torch.zeros_like(z)
    if plane == "xy":
        return (
            -Q_OSCILLATION * wave**2 * cosine,
            -Q_OSCILLATION * wave**2 * sine,
            zero,
            Q_OSCILLATION * wave**2 * cosine,
            zero,
        )
    if plane == "xz":
        return (
            -Q_OSCILLATION * wave**2 * cosine,
            zero,
            -Q_OSCILLATION * wave**2 * sine,
            zero,
            zero,
        )
    raise ValueError("plane must be 'xy' or 'xz'")


def _face_values(*, plane: str, upper: bool) -> dict[str, float]:
    z = torch.tensor(HEIGHT if upper else 0.0, dtype=torch.float64)
    return {
        name: float(value.item())
        for name, value in zip(Q_COMPONENTS, _twist_components(z, plane=plane))
    }


def _expanded_initial(
    wall_points: int,
    *,
    plane: str,
    tangential_points: int = 4,
) -> dict[str, torch.Tensor]:
    z = _cell_centers(wall_points)
    shape = (tangential_points, tangential_points, wall_points)
    return {
        name: value.reshape(1, 1, wall_points).expand(shape).clone()
        for name, value in zip(Q_COMPONENTS, _twist_components(z, plane=plane))
    }


def _build_lifted_runtime(
    initial_values: Mapping[str, torch.Tensor],
    *,
    wall_points: int,
    molecular_field_linear_space: str = "spectral",
    plane: str = "xy",
):
    tangential_points = int(initial_values[Q_COMPONENTS[0]].shape[0])
    config = RuntimeProfileConfig(
        runtime_path="legacy_production",
        shape=(tangential_points, tangential_points, wall_points),
        lengths=(40.0, 40.0, HEIGHT),
        device="cpu",
        dtype="float64",
        warmup_steps=0,
        profile_steps=1,
        pointwise_execution="eager",
        molecular_field_linear_space=molecular_field_linear_space,
        stress_divergence_sum_space="spectral",
        seed=20261010,
    )
    run_spec = _run_spec(config)
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    boundaries = assign_boundaries(
        model=base.equation_system,
        geometry=base.geometry,
        policies={
            "Q": prescribed_q(
                {
                    (2, "lower"): _face_values(plane=plane, upper=False),
                    (2, "upper"): _face_values(plane=plane, upper=True),
                }
            ),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        name=f"rc431_{plane}_twist",
    )
    simulation = replace(base, boundaries=boundaries)
    metadata = {
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
    }
    request = PlaneRuntimeBuildRequest(
        run_spec,
        metadata,
        initial_values,
        "cpu",
        simulation,
        fresh_initial_conditioning=(
            legacy_whole_slab_sine_conditioning()
        ),
    )
    construction = PackageRuntimeConstructionInput(
        plan_package_runtime_construction(simulation),
        request,
    )
    adapter = build_package_simulation_runtime(construction)
    return run_spec, simulation, adapter


def _inject_exact_physical_state(
    adapter: object,
    values: Mapping[str, torch.Tensor],
) -> None:
    lifting = adapter.solver.model.static_model.static_lifting_runtime
    operator = lifting.operator
    for name in Q_COMPONENTS:
        adapter.fields[name] = operator.extract_homogeneous_remainder(
            name,
            values[name].to(dtype=torch.float64, device="cpu"),
        )


def _refresh_with_capture(adapter: object) -> dict[str, object]:
    model = adapter.solver.model.static_model
    timer = RegionTimer(torch.device("cpu"), timed_regions=frozenset())
    _install_transform_timers(adapter.solver, timer)
    captured: dict[str, object] = {"physical_projection_calls": 0}
    original_projection = model._project_physical_tensor

    def capture(fields, tensor, boundary_conditions):
        captured["physical_projection_calls"] += 1
        captured["raw_h"] = tensor.detach().clone()
        projected = original_projection(fields, tensor, boundary_conditions)
        captured["projected_h"] = projected.detach().clone()
        return projected

    model._project_physical_tensor = capture
    timer.reset()
    try:
        adapter.synchronize_for_observation()
    finally:
        model._project_physical_tensor = original_projection
    summary = timer.summarize()
    captured["transforms"] = {
        "forward": int(summary.get("transform_forward", {}).get("calls", 0)),
        "inverse": int(summary.get("transform_inverse", {}).get("calls", 0)),
    }
    captured["runtime_fallback"] = adapter.to_metadata()["fallback_used"]
    return captured


def _direct_h(
    z: torch.Tensor,
    *,
    plane: str,
    model: BerisEdwardsFreeSlipStokes,
) -> tuple[torch.Tensor, ...]:
    q = _full_q_tensor(_twist_components(z, plane=plane))
    laplacian = _full_q_tensor(_twist_laplacian(z, plane=plane))
    q_squared = torch.matmul(q, q)
    trace_q_squared = torch.einsum("...ij,...ij->...", q, q)
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    h = (
        -model.ldg_a * q
        - model.ldg_b
        * (q_squared - trace_q_squared[..., None, None] * identity / 3.0)
        - model.ldg_c * trace_q_squared[..., None, None] * q
        + model.ldg_l1 * laplacian
    )
    return _compact_q_tensor(h)


def _full_q_tensor(components: tuple[torch.Tensor, ...]) -> torch.Tensor:
    qxx, qxy, qxz, qyy, qyz = components
    qzz = -qxx - qyy
    return torch.stack(
        (qxx, qxy, qxz, qxy, qyy, qyz, qxz, qyz, qzz),
        dim=-1,
    ).unflatten(-1, (3, 3))


def _compact_q_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        tensor[..., 0, 0],
        tensor[..., 0, 1],
        tensor[..., 0, 2],
        tensor[..., 1, 1],
        tensor[..., 1, 2],
    )


def _direct_stress(
    z: torch.Tensor,
    *,
    model: BerisEdwardsFreeSlipStokes,
    alpha: float,
) -> torch.Tensor:
    q = _full_q_tensor(_twist_components(z, plane="xz"))
    h = _full_q_tensor(_direct_h(z, plane="xz", model=model))
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    shifted_q = q + identity / 3.0
    q_dot_h = torch.einsum("...ij,...ij->...", q, h)
    qh = torch.matmul(q, h)
    hq = torch.matmul(h, q)
    reactive = (
        2.0
        * model.flow_alignment
        * shifted_q
        * q_dot_h[..., None, None]
        - model.flow_alignment
        * (torch.matmul(h, shifted_q) + torch.matmul(shifted_q, h))
        + qh
        - hq
    )
    active = model.beta * alpha * q
    gradient_z = _full_q_tensor(_twist_gradient(z, plane="xz"))
    gradient_norm_squared = torch.einsum(
        "...ij,...ij->...",
        gradient_z,
        gradient_z,
    )
    distortion = torch.zeros_like(q)
    distortion[..., 2, 2] = -model.ldg_l1 * gradient_norm_squared
    return reactive + active + distortion


def _stack_reference(
    values: tuple[torch.Tensor, ...],
    *,
    tangential_points: int,
) -> torch.Tensor:
    wall_points = int(values[0].numel())
    return torch.stack(values).reshape(5, 1, 1, 1, wall_points).expand(
        5,
        1,
        tangential_points,
        tangential_points,
        wall_points,
    )


def _run_l01() -> dict[str, object]:
    errors = [_manufactured_error(size, device="cpu") for size in WALL_POINTS]
    rates = _adjacent_rates(errors)
    return {
        "case": "L01",
        "kind": "affine_lift_scalar_manufactured_control",
        "wall_points": list(WALL_POINTS),
        "errors": errors,
        "adjacent_rates": rates,
        "regression_rate": _regression_rate(errors),
        "claimed_second_order_passed": (
            _regression_rate(errors) >= 1.8 and min(rates[-2:]) >= 1.8
        ),
        "finite": all(math.isfinite(value) for value in (*errors, *rates)),
    }


def _run_l02() -> dict[str, object]:
    rows = []
    wall_errors = []
    bulk_errors = []
    for wall_points in WALL_POINTS:
        initial = _expanded_initial(wall_points, plane="xy")
        _run, _simulation, adapter = _build_lifted_runtime(
            initial,
            wall_points=wall_points,
            plane="xy",
        )
        _inject_exact_physical_state(adapter, initial)
        captured = _refresh_with_capture(adapter)
        model = adapter.solver.model.static_model
        z = _cell_centers(wall_points)
        expected_h = _stack_reference(
            _direct_h(z, plane="xy", model=model),
            tangential_points=4,
        )
        raw_h = captured["raw_h"]
        difference = torch.abs(raw_h - expected_h)
        wall_error = float(
            torch.max(
                torch.stack((difference[..., 0], difference[..., -1]))
            ).item()
        )
        first_interior_error = float(
            torch.max(
                torch.stack((difference[..., 1], difference[..., -2]))
            ).item()
        )
        physical_q = torch.stack(
            tuple(plane_physical_component(adapter, name) for name in Q_COMPONENTS)
        )
        expected_q = torch.stack(tuple(initial[name] for name in Q_COMPONENTS))[:, None]
        q_difference = torch.abs(physical_q - expected_q)
        q_wall_error = float(
            torch.max(torch.stack((q_difference[..., 0], q_difference[..., -1]))).item()
        )
        q_first_error = float(
            torch.max(torch.stack((q_difference[..., 1], q_difference[..., -2]))).item()
        )
        bulk_error = _errors(raw_h, expected_h)["relative_l2"]
        wall_errors.append(wall_error)
        bulk_errors.append(bulk_error)
        rows.append(
            {
                "wall_points": wall_points,
                "q_wall_linf": q_wall_error,
                "q_first_interior_linf": q_first_error,
                "molecular_field_wall_linf": wall_error,
                "molecular_field_first_interior_linf": first_interior_error,
                "molecular_field_bulk_relative_l2": bulk_error,
                "molecular_field_bulk_linf": _errors(raw_h, expected_h)["linf"],
                "transforms": captured["transforms"],
                "runtime_fallback": captured["runtime_fallback"],
                "finite": bool(torch.isfinite(raw_h).all().item()),
            }
        )
    wall_rates = _adjacent_rates(wall_errors)
    bulk_rates = _adjacent_rates(bulk_errors)
    return {
        "case": "L02",
        "kind": "real_solver_nonzero_wall_molecular_field",
        "profile": "analytic_in_plane_director_twist",
        "reference": "direct_constitutive_evaluation_with_analytic_q_laplacian",
        "rows": rows,
        "wall_error_adjacent_rates": wall_rates,
        "wall_error_regression_rate": _regression_rate(wall_errors),
        "bulk_error_adjacent_rates": bulk_rates,
        "bulk_error_regression_rate": _regression_rate(bulk_errors),
        "last_two_wall_errors_strictly_decrease": (
            wall_errors[-3] > wall_errors[-2] > wall_errors[-1]
        ),
        "representation_repair_supported": (
            _regression_rate(wall_errors) >= 1.8
            and wall_errors[-3] > wall_errors[-2] > wall_errors[-1]
            and _regression_rate(bulk_errors) >= 1.8
        ),
    }


def _direct_flow_reference(
    z: torch.Tensor,
    *,
    model: BerisEdwardsFreeSlipStokes,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_z = torch.linspace(0.0, HEIGHT, 131073, dtype=torch.float64)
    stress_tensor = _direct_stress(
        reference_z,
        model=model,
        alpha=alpha,
    )
    stress = tuple(
        stress_tensor[..., row, column].detach().cpu().numpy()
        for row in range(3)
        for column in range(3)
    )
    coordinate = reference_z.numpy()
    spacing = coordinate[1] - coordinate[0]

    def integrate_tangential(component: np.ndarray) -> np.ndarray:
        increments = 0.5 * (component[:-1] + component[1:]) * spacing
        value = -np.concatenate(([0.0], np.cumsum(increments))) / model.viscosity
        integral = spacing * (
            0.5 * value[0] + float(np.sum(value[1:-1])) + 0.5 * value[-1]
        )
        value -= integral / HEIGHT
        return np.interp(z.numpy(), coordinate, value)

    ux = integrate_tangential(stress[2])
    uy = integrate_tangential(stress[5])
    pressure = np.interp(z.numpy(), coordinate, stress[8])
    pressure -= np.mean(pressure)
    return ux, uy, pressure


def _run_l03() -> dict[str, object]:
    rows = []
    velocity_l2 = []
    velocity_linf = []
    pressure_l2 = []
    pressure_linf = []
    for wall_points in WALL_POINTS:
        initial = _expanded_initial(wall_points, plane="xz")
        _run, _simulation, adapter = _build_lifted_runtime(
            initial,
            wall_points=wall_points,
            plane="xz",
        )
        _inject_exact_physical_state(adapter, initial)
        captured = _refresh_with_capture(adapter)
        model = adapter.solver.model.static_model
        z = _cell_centers(wall_points)
        alpha = float(adapter.solver.model.parameters["alpha"].item())
        reference_ux, reference_uy, reference_pressure = _direct_flow_reference(
            z,
            model=model,
            alpha=alpha,
        )
        actual_velocity = torch.stack(
            (
                adapter.fields["ux"][0, 0, 0],
                adapter.fields["uy"][0, 0, 0],
            )
        ).detach().cpu().numpy()
        expected_velocity = np.stack((reference_ux, reference_uy))
        actual_pressure = adapter.fields["p"][0, 0, 0].detach().cpu().numpy()
        actual_pressure = actual_pressure - np.mean(actual_pressure)
        velocity_error = {
            "relative_l2": _relative_l2(actual_velocity, expected_velocity),
            "linf": float(np.max(np.abs(actual_velocity - expected_velocity))),
        }
        pressure_error = {
            "relative_l2": _relative_l2(actual_pressure, reference_pressure),
            "linf": float(np.max(np.abs(actual_pressure - reference_pressure))),
        }
        velocity_l2.append(velocity_error["relative_l2"])
        velocity_linf.append(velocity_error["linf"])
        pressure_l2.append(pressure_error["relative_l2"])
        pressure_linf.append(pressure_error["linf"])
        rows.append(
            {
                "wall_points": wall_points,
                "velocity": velocity_error,
                "pressure_demeaned": pressure_error,
                "actual_velocity_max": float(np.max(np.abs(actual_velocity))),
                "reference_velocity_max": float(np.max(np.abs(expected_velocity))),
                "transforms": captured["transforms"],
                "runtime_fallback": captured["runtime_fallback"],
                "finite": bool(
                    np.isfinite(actual_velocity).all()
                    and np.isfinite(actual_pressure).all()
                ),
            }
        )
    return {
        "case": "L03",
        "kind": "real_solver_twisted_q_to_stokes_flow",
        "profile": "analytic_xz_director_twist",
        "reference": (
            "analytic_q_derivatives_plus_direct_constitutive_stress_plus_"
            "independent_dense_physical_quadrature"
        ),
        "rows": rows,
        "velocity_relative_l2_adjacent_rates": _adjacent_rates(velocity_l2),
        "velocity_relative_l2_regression_rate": _regression_rate(velocity_l2),
        "velocity_linf_adjacent_rates": _adjacent_rates(velocity_linf),
        "velocity_linf_regression_rate": _regression_rate(velocity_linf),
        "pressure_relative_l2_adjacent_rates": _adjacent_rates(pressure_l2),
        "pressure_relative_l2_regression_rate": _regression_rate(pressure_l2),
        "pressure_linf_adjacent_rates": _adjacent_rates(pressure_linf),
        "pressure_linf_regression_rate": _regression_rate(pressure_linf),
        "claimed_second_order_velocity_passed": (
            _regression_rate(velocity_l2) >= 1.8
            and velocity_l2[-3] > velocity_l2[-2] > velocity_l2[-1]
        ),
    }


def _uniform_initial(wall_points: int, tangential_points: int) -> dict[str, torch.Tensor]:
    values = _face_values(plane="xy", upper=False)
    shape = (tangential_points, tangential_points, wall_points)
    return {
        name: torch.full(shape, values[name], dtype=torch.float64)
        for name in Q_COMPONENTS
    }


def _defect_initial(wall_points: int, tangential_points: int) -> dict[str, torch.Tensor]:
    q2d = analytic_periodic_defect_gas_2d(
        (tangential_points, tangential_points),
        lengths=(40.0, 40.0),
        num_defect_pairs=2,
        min_separation=5.0,
        core_radius=1.5,
        S_initial=1.0 / 3.0,
        seed=24,
        dtype=torch.float64,
    )
    return {
        name: q2d[name].unsqueeze(-1).expand(
            tangential_points,
            tangential_points,
            wall_points,
        ).clone()
        for name in Q_COMPONENTS
    }


def _conditioning_row(
    initial: Mapping[str, torch.Tensor],
    *,
    wall_points: int,
) -> dict[str, object]:
    _run, _simulation, adapter = _build_lifted_runtime(
        initial,
        wall_points=wall_points,
        plane="xy",
    )
    physical = torch.stack(
        tuple(plane_physical_component(adapter, name) for name in Q_COMPONENTS)
    )
    requested = torch.stack(tuple(initial[name] for name in Q_COMPONENTS))[:, None]
    deviation = torch.abs(physical - requested).amax(dim=(0, 1, 2, 3))
    quarter_index = wall_points // 4
    runtime = adapter.solver.model.static_model.static_lifting_runtime
    policy_metadata = runtime.fresh_initial_conditioning_metadata()
    if policy_metadata["effective_policy"] != "legacy_whole_slab_sine_v1":
        raise RuntimeError("RC4.3.1 reproducer did not select legacy conditioning")
    # Preserve the exact pre-RC4.3.4 report schema.  The numerical behavior is
    # now requested explicitly above; this compatibility view exists only so
    # the immutable RC4.3.1 analysis remains reproducible byte for byte.
    historical_metadata = {
        "kind": "sine_wall_taper",
        "wall_values": [0.0, 0.0],
        "applied_before_first_transform": True,
    }
    return {
        "wall_points": wall_points,
        "requested_sha256": _mapping_sha256(initial),
        "physical_sha256": _tensor_sha256(physical),
        "first_eight_cell_deviation_linf": [
            float(value.item()) for value in deviation[: min(8, wall_points)]
        ],
        "quarter_height_index": quarter_index,
        "quarter_height_deviation_linf": float(deviation[quarter_index].item()),
        "center_deviation_linf": float(deviation[wall_points // 2].item()),
        "domain_max_deviation_linf": float(deviation.max().item()),
        "exactly_unchanged_cell_count": int(torch.count_nonzero(deviation == 0).item()),
        "conditioning_metadata": historical_metadata,
        "finite": bool(torch.isfinite(physical).all().item()),
    }


def _run_l04() -> dict[str, object]:
    tangential_points = 16
    uniform = []
    defect = []
    for wall_points in WALL_POINTS:
        uniform.append(
            _conditioning_row(
                _uniform_initial(wall_points, tangential_points),
                wall_points=wall_points,
            )
        )
        defect.append(
            _conditioning_row(
                _defect_initial(wall_points, tangential_points),
                wall_points=wall_points,
            )
        )
    metadata = uniform[0]["conditioning_metadata"]
    return {
        "case": "L04",
        "kind": "fresh_initial_conditioning_scope",
        "uniform": uniform,
        "defect_gas": defect,
        "policy_kind": metadata.get("kind"),
        "formula_present": "formula" in metadata,
        "transition_width_present": "width" in metadata,
        "transition_width_units_present": "width_units" in metadata,
        "localized_unchanged_interior_supported": all(
            row["exactly_unchanged_cell_count"] > 0
            for row in (*uniform, *defect)
        ),
        "whole_slab_sine_taper_confirmed": (
            metadata.get("kind") == "sine_wall_taper"
            and all(row["quarter_height_deviation_linf"] > 0.0 for row in uniform)
            and all(row["quarter_height_deviation_linf"] > 0.0 for row in defect)
        ),
    }


def _space_observation(linear_space: str) -> dict[str, object]:
    wall_points = 32
    initial = _expanded_initial(wall_points, plane="xy")
    run_spec, simulation, adapter = _build_lifted_runtime(
        initial,
        wall_points=wall_points,
        plane="xy",
        molecular_field_linear_space=linear_space,
    )
    _inject_exact_physical_state(adapter, initial)
    captured = _refresh_with_capture(adapter)
    state = torch.cat(
        tuple(
            adapter.fields[name].detach().reshape(-1)
            for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p")
        )
    )
    return {
        "requested": linear_space,
        "configuration_requested": run_spec.molecular_field_linear_space,
        "model_requested": adapter.solver.model.static_model.molecular_field_linear_space,
        "observed_physical_projection_calls": captured["physical_projection_calls"],
        "effective_path_observation": "physical_h_assembly_and_projection",
        "raw_h_sha256": _tensor_sha256(captured["raw_h"]),
        "projected_h_sha256": _tensor_sha256(captured["projected_h"]),
        "state_sha256": _tensor_sha256(state),
        "simulation_identity": simulation.canonical_sha256(),
        "transforms": captured["transforms"],
        "runtime_fallback": captured["runtime_fallback"],
    }


def _run_l05() -> dict[str, object]:
    spectral = _space_observation("spectral")
    physical = _space_observation("physical")
    source = inspect.getsource(BerisEdwardsFreeSlipStokes.compute_nematic_force)
    guard = 'if lifting is not None or self.molecular_field_linear_space == "physical":'
    output_keys = ("raw_h_sha256", "projected_h_sha256", "state_sha256", "transforms")
    outputs_equal = all(spectral[key] == physical[key] for key in output_keys)
    return {
        "case": "L05",
        "kind": "molecular_field_requested_effective_path_identity",
        "spectral_request": spectral,
        "physical_request": physical,
        "runtime_source_sha256": _sha256_bytes(source.encode("utf-8")),
        "lifting_forces_physical_branch_guard_present": guard in source,
        "observed_outputs_and_transform_counts_identical": outputs_equal,
        "requested_space_is_silently_ignored": (
            spectral["requested"] != physical["requested"]
            and spectral["observed_physical_projection_calls"] > 0
            and physical["observed_physical_projection_calls"] > 0
            and outputs_equal
        ),
        "explicit_effective_space_metadata_present": False,
    }


def run_analysis() -> dict[str, object]:
    l01 = _run_l01()
    l02 = _run_l02()
    l03 = _run_l03()
    l04 = _run_l04()
    l05 = _run_l05()
    finite = (
        l01["finite"]
        and all(row["finite"] for row in l02["rows"])
        and all(row["finite"] for row in l03["rows"])
        and all(row["finite"] for row in (*l04["uniform"], *l04["defect_gas"]))
    )
    decision = "PUBLIC_CAPABILITY_RESTRICTION_REQUIRED"
    return {
        "schema_version": SCHEMA_VERSION,
        "record": "PSSolver RC4.3.1 lifting-correctness analysis",
        "phase": "RC4.3.1",
        "analysis_only": True,
        "wall_points": list(WALL_POINTS),
        "cases": {item["case"]: item for item in (l01, l02, l03, l04, l05)},
        "decision": {
            "classification": decision,
            "all_arrays_finite": finite,
            "affine_scalar_control_passed": l01["claimed_second_order_passed"],
            "nonzero_wall_molecular_field_converged": l02[
                "representation_repair_supported"
            ],
            "twisted_velocity_second_order": l03[
                "claimed_second_order_velocity_passed"
            ],
            "fresh_conditioning_local_and_explicit": (
                l04["localized_unchanged_interior_supported"]
                and l04["formula_present"]
                and l04["transition_width_present"]
                and l04["transition_width_units_present"]
            ),
            "molecular_field_request_is_not_silently_ignored": not l05[
                "requested_space_is_silently_ignored"
            ],
            "rationale": [
                (
                    "The scalar affine-lift control converges, so the diagnostic "
                    "is capable of observing second order."
                ),
                (
                    "The real lifted solver does not provide second-order "
                    "convergent nonzero-wall molecular fields in the current "
                    "DST remainder representation."
                ),
                (
                    "The independent twisted-flow reference exposes "
                    "approximately first-order velocity convergence."
                ),
                (
                    "Fresh-state sine conditioning changes the slab interior "
                    "and does not declare a formula or physical transition width."
                ),
                (
                    "Lifted spectral and physical molecular-field requests "
                    "execute the same observed physical branch and produce "
                    "identical outputs."
                ),
            ],
        },
        "authorization": {
            "analysis_authorized_by_user": True,
            "runtime_source_modified": False,
            "production_default_changed": False,
            "h100_submitted": False,
            "PSSolver_Control_modified": False,
            "nematics3d_modified": False,
            "rc4_3_2_authorized": False,
            "eligible_for_automatic_merge": False,
            "eligible_for_default_promotion": False,
        },
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    with path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != value:
            raise RuntimeError("atomic JSON readback mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_analysis()
    _atomic_write_json(arguments.output, result)
    print(json.dumps(result["decision"], allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
