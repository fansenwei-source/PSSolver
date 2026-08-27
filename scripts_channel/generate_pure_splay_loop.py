#!/usr/bin/env python3
"""Generate a solver-ready ideal pure-splay disclination loop in a channel.

Calculation method
------------------
The construction follows the ideal initial-loop model in
``3D Dry Uniaxial Active Nematics in Bulk``.  In the paper's principal-plane
convention, N, M, and L are the decreasing-eigenvalue eigenvectors of
``q_box = <n n>_box``; the N-M principal plane has normal L.  For an ideal
initial loop, the paper additionally assumes ``k || nu`` and ``Omega || L``.

This generator fixes

    N = k = nu = ex,  M = ey (or ez),  L = N x M,  Omega = L,

which classifies the complete pi-rotation, one-dimensional skyrmion crossing
the loop as pure splay.  This is deliberately different from the paper's
small-perturbation mapping: for a small perturbation, k parallel N is bend;
for the ideal loop skyrmion, k parallel N is splay.  The loop lies in the M-L
plane, while N-M is the principal plane.

For a loop centered at c with radius R,
each cell-centered position x is decomposed into

    ell = periodic_minimum_image(x_x - c_x),
    rho = sqrt((x_y - c_y)^2 + (x_z - c_z)^2).

The defect ring is the set ell = 0 and rho = R.  With h denoting the disturbed
half-thickness along k, the paper's simplest angle profiles are

    theta = pi*ell/(2*h) + pi/2                    if rho < R and |ell| <= h,
    theta = -epsilon(rho)*sin(pi*ell/h)            if rho >= R and |ell| <= h,
    theta = 0                                      otherwise,

where

    epsilon(rho) = epsilon0*exp(-(rho - R)^2/(2*lambda_epsilon^2)).

Away from the defect core, the director is
n = cos(theta)*N + sin(theta)*M.  Directly joining the inside and outside
angle branches at rho = R would create a nonphysical jump on a cylindrical
surface.  The script therefore interpolates the two uniaxial Q tensors across
a narrow radial transition layer.  This tensor interpolation permits a
biaxial core and avoids choosing a director where it is undefined.

The core amplitude is additionally regularized using

    d_core = sqrt((rho - R)^2 + ell^2),
    Q <- tanh(d_core/a_core)*Q.

S_bulk is either supplied by the user or obtained from the positive
uniaxial bulk-equilibrium root

    3*c_Q*S_bulk^2 + b_Q*S_bulk + 2*a_Q = 0.

Finally, Q is smoothly blended to the uniform x-aligned background in buffers
next to the y and z walls.  This preserves the channel's homogeneous Neumann
boundary representation while the x direction remains periodic.  The output
Q array stores (Qxx, Qxy, Qxz, Qyy, Qyz) on its final axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pssolver.loop_conventions import convention_metadata
from pssolver.models.active_nematics import (
    Q_components,
    Q_convention_metadata,
    S_from_Q,
    positive_equilibrium_S,
    uniaxial_Q,
)


DEFAULT_SHAPE = (512, 40, 40)
DEFAULT_LENGTHS = (128.0, 10.0, 10.0)
Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")




def cell_centered_axes(
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for count, length in zip(shape, lengths):
        spacing = length / count
        axes.append((np.arange(count, dtype=float) + 0.5) * spacing)
    return tuple(axes)


def periodic_minimum_image(values: np.ndarray, period: float) -> np.ndarray:
    return (values + 0.5 * period) % period - 0.5 * period


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def wall_blend_weight(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    wall_buffer: float,
) -> np.ndarray:
    """Return a weight that is zero at y/z boundary cells and one in the bulk."""
    if wall_buffer <= 0.0:
        return np.ones(shape, dtype=float)

    wall_distances = []
    for axis in (1, 2):
        indices = np.arange(shape[axis], dtype=float)
        distance = np.minimum(indices, shape[axis] - 1.0 - indices) * spacing[axis]
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        wall_distances.append(np.broadcast_to(distance.reshape(reshape), shape))

    distance = np.minimum.reduce(wall_distances)
    return smoothstep(distance / wall_buffer)


def q5_from_director(
    director: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    components = Q_components(uniaxial_Q(director, S))
    return np.stack(
        tuple(components[name] for name in Q_COMPONENTS),
        axis=-1,
    ).astype(np.float32)


def q5_to_matrix(q5: np.ndarray) -> np.ndarray:
    matrix = np.empty(q5.shape[:-1] + (3, 3), dtype=q5.dtype)
    matrix[..., 0, 0] = q5[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = q5[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = q5[..., 2]
    matrix[..., 1, 1] = q5[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = q5[..., 4]
    matrix[..., 2, 2] = -q5[..., 0] - q5[..., 3]
    return matrix


def principal_director_and_S(q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, eigenvectors = np.linalg.eigh(q5_to_matrix(q5))
    director = eigenvectors[..., :, -1]
    S = np.asarray(S_from_Q(q5))
    return director.astype(np.float32), S.astype(np.float32)


def defect_plaquette_mask(
    director: np.ndarray,
    axis_a: int,
    axis_b: int,
) -> np.ndarray:
    """Detect nematic half-winding on plaquettes; x is periodic, y/z are not."""
    n00 = director
    n10 = np.roll(director, -1, axis=axis_a)
    n01 = np.roll(director, -1, axis=axis_b)
    n11 = np.roll(n10, -1, axis=axis_b)
    link_product = (
        np.sum(n00 * n10, axis=-1)
        * np.sum(n10 * n11, axis=-1)
        * np.sum(n11 * n01, axis=-1)
        * np.sum(n01 * n00, axis=-1)
    )
    valid = [slice(None), slice(None), slice(None)]
    for axis in (axis_a, axis_b):
        if axis != 0:
            valid[axis] = slice(0, -1)
    return link_product[tuple(valid)] < 0.0


def construct_loop(
    *,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    center: tuple[float, float, float],
    radius: float,
    m_axis_name: str,
    half_thickness: float,
    epsilon0: float,
    epsilon_decay_length: float,
    core_size: float,
    radial_transition_size: float,
    S_bulk: float,
    wall_buffer: float,
) -> dict[str, np.ndarray]:
    axes = cell_centered_axes(shape, lengths)
    spacing = tuple(length / count for count, length in zip(shape, lengths))
    x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)

    ell = np.broadcast_to(
        periodic_minimum_image(x - center[0], lengths[0]),
        shape,
    )
    dy = y - center[1]
    dz = z - center[2]
    rho = np.broadcast_to(np.sqrt(dy * dy + dz * dz), shape)
    distance_to_ring = np.sqrt((rho - radius) ** 2 + ell * ell)

    disturbed = np.abs(ell) <= half_thickness
    inside_angle = np.zeros(shape, dtype=float)
    inside_angle[disturbed] = 0.5 * np.pi * (
        ell[disturbed] / half_thickness + 1.0
    )

    epsilon = epsilon0 * np.exp(
        -0.5 * ((rho - radius) / epsilon_decay_length) ** 2
    )
    outside_angle = np.zeros(shape, dtype=float)
    outside_angle[disturbed] = -epsilon[disturbed] * np.sin(
        np.pi * ell[disturbed] / half_thickness
    )

    axis_n = np.array((1.0, 0.0, 0.0))
    axis_m = (
        np.array((0.0, 1.0, 0.0))
        if m_axis_name == "y"
        else np.array((0.0, 0.0, 1.0))
    )
    axis_l = np.cross(axis_n, axis_m)
    director_inside = (
        np.cos(inside_angle)[..., None] * axis_n
        + np.sin(inside_angle)[..., None] * axis_m
    )
    director_outside = (
        np.cos(outside_angle)[..., None] * axis_n
        + np.sin(outside_angle)[..., None] * axis_m
    )
    uniform_S = np.full(shape, S_bulk)
    q_inside = q5_from_director(director_inside, uniform_S)
    q_outside = q5_from_director(director_outside, uniform_S)

    inside_weight = 0.5 * (
        1.0 - np.tanh((rho - radius) / radial_transition_size)
    )
    q_orientation = (
        inside_weight[..., None] * q_inside
        + (1.0 - inside_weight[..., None]) * q_outside
    )
    q_orientation[~disturbed] = q_outside[~disturbed]
    core_suppression = np.tanh(distance_to_ring / core_size)
    q_model = q_orientation * core_suppression[..., None]

    background_director = np.broadcast_to(axis_n, shape + (3,))
    background_S = np.full(shape, S_bulk)
    q_background = q5_from_director(background_director, background_S)
    blend_weight = wall_blend_weight(shape, spacing, wall_buffer)
    q_output = (
        blend_weight[..., None] * q_model
        + (1.0 - blend_weight[..., None]) * q_background
    ).astype(np.float32)
    director_output, S_output = principal_director_and_S(q_output)

    return {
        "Q": q_output,
        "director_model": director_output,
        "S_model": S_output,
        "director_defined_mask": S_output >= 2.0 / 30.0,
        "inside_angle": inside_angle.astype(np.float32),
        "outside_angle": outside_angle.astype(np.float32),
        "inside_weight": inside_weight.astype(np.float32),
        "wall_blend_weight": blend_weight.astype(np.float32),
        "distance_to_ring": distance_to_ring.astype(np.float32),
        "N": axis_n,
        "M": axis_m,
        "L": axis_l,
    }


def validate_inputs(args: argparse.Namespace) -> None:
    shape = tuple(args.shape)
    lengths = tuple(args.lengths)
    center = tuple(args.center)
    if any(count <= 1 for count in shape):
        raise ValueError(f"shape entries must exceed one, got {shape}")
    if any(length <= 0.0 for length in lengths):
        raise ValueError(f"lengths must be positive, got {lengths}")
    if any(not 0.0 <= coordinate < length for coordinate, length in zip(center, lengths)):
        raise ValueError(f"center {center} must lie inside domain {lengths}")
    for name in ("radius", "half_thickness", "core_size"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive, got {getattr(args, name)}")
    if args.epsilon_deg < 0.0:
        raise ValueError(f"epsilon_deg must be nonnegative, got {args.epsilon_deg}")
    if args.epsilon_decay_length < 0.0:
        raise ValueError(
            "epsilon_decay_length must be zero (automatic) or positive, got "
            f"{args.epsilon_decay_length}"
        )
    if args.radial_transition_size < 0.0:
        raise ValueError(
            "radial_transition_size must be zero (automatic) or positive, got "
            f"{args.radial_transition_size}"
        )
    if args.wall_buffer < 0.0:
        raise ValueError(f"wall_buffer must be nonnegative, got {args.wall_buffer}")
    if args.half_thickness >= 0.5 * lengths[0]:
        raise ValueError("half_thickness must be smaller than half the x period")

    wall_clearance = min(
        center[1],
        lengths[1] - center[1],
        center[2],
        lengths[2] - center[2],
    )
    required_clearance = args.radius + args.core_size + args.wall_buffer
    if required_clearance >= wall_clearance:
        raise ValueError(
            "the loop core overlaps the wall blending region: "
            f"required clearance={required_clearance}, available={wall_clearance}. "
            "Reduce radius/core-size/wall-buffer or move the center."
        )


def validation_summary(
    fields: dict[str, np.ndarray],
    *,
    S_bulk: float,
) -> dict[str, object]:
    q5 = fields["Q"]
    S = fields["S_model"]
    director = fields["director_model"]
    director_norm = np.linalg.norm(director, axis=-1)
    disturbed = (np.abs(fields["inside_angle"]) > 1e-6) | (
        np.abs(fields["outside_angle"]) > 1e-6
    )
    q_box = np.einsum(
        "ni,nj->ij", director[disturbed], director[disturbed]
    ) / np.count_nonzero(disturbed)
    q_box_values, q_box_axes = np.linalg.eigh(q_box)
    order = np.argsort(q_box_values)[::-1]
    q_box_values = q_box_values[order]
    q_box_axes = q_box_axes[:, order]
    expected_axes = np.column_stack((fields["N"], fields["M"], fields["L"]))
    q_box_alignment = np.abs(expected_axes.T @ q_box_axes)
    wall_delta = np.concatenate(
        (
            (q5[:, 1, :, :] - q5[:, 0, :, :]).reshape(-1),
            (q5[:, -1, :, :] - q5[:, -2, :, :]).reshape(-1),
            (q5[:, :, 1, :] - q5[:, :, 0, :]).reshape(-1),
            (q5[:, :, -1, :] - q5[:, :, -2, :]).reshape(-1),
        )
    )
    defect_counts = {
        "xy": int(np.count_nonzero(defect_plaquette_mask(fields["director_model"], 0, 1))),
        "xz": int(np.count_nonzero(defect_plaquette_mask(fields["director_model"], 0, 2))),
        "yz": int(np.count_nonzero(defect_plaquette_mask(fields["director_model"], 1, 2))),
    }
    return {
        "finite_Q": bool(np.isfinite(q5).all()),
        "Q_shape": list(q5.shape),
        "Q_components": list(Q_COMPONENTS),
        "max_director_norm_error": float(np.max(np.abs(director_norm - 1.0))),
        "S_min": float(np.min(S)),
        "S_median": float(np.median(S)),
        "S_bulk": S_bulk,
        "q_box_definition": "<n n> over the constructed disturbed region",
        "q_box_eigenvalues_descending": q_box_values.tolist(),
        "q_box_abs_alignment_expected_NML": q_box_alignment.tolist(),
        "q_box_NML_consistent": bool(
            np.all(np.diag(q_box_alignment) > 1.0 - 1e-5)
        ),
        "max_adjacent_wall_cell_Q_difference": float(np.max(np.abs(wall_delta))),
        "wall_Q_is_exact_background": bool(
            np.all(fields["wall_blend_weight"][:, (0, -1), :] == 0.0)
            and np.all(fields["wall_blend_weight"][:, :, (0, -1)] == 0.0)
        ),
        "defect_plaquette_counts": defect_counts,
        "yz_plane_topology": bool(
            defect_counts["xy"] > 0
            and defect_counts["xz"] > 0
            and defect_counts["yz"] == 0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("pure_splay_loop"))
    parser.add_argument("--shape", type=int, nargs=3, default=DEFAULT_SHAPE)
    parser.add_argument("--lengths", type=float, nargs=3, default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=(64.0, 5.0, 5.0),
        help="Loop center in physical channel coordinates.",
    )
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument(
        "--m-axis",
        choices=("y", "z"),
        default="y",
        help=(
            "Principal perturbation axis M. For this ideal pure-splay loop, "
            "N, k, and nu are +x; L = N x M and Omega = L."
        ),
    )
    parser.add_argument("--half-thickness", type=float, default=2.5)
    parser.add_argument("--epsilon-deg", type=float, default=20.0)
    parser.add_argument(
        "--epsilon-decay-length",
        type=float,
        default=0.0,
        help="Physical decay length; zero uses the loop radius.",
    )
    parser.add_argument("--core-size", type=float, default=0.5)
    parser.add_argument(
        "--radial-transition-size",
        type=float,
        default=0.0,
        help="Width of the inside/outside Q interpolation; zero uses core-size.",
    )
    parser.add_argument(
        "--bulk-s",
        dest="S_bulk",
        type=float,
        default=None,
        help=(
            "Positive fixed S in Q=(3S/2)(nn-I/3); omitted derives S "
            "from aQ, bQ, cQ."
        ),
    )
    parser.add_argument("--aQ", type=float, default=-1.0)
    parser.add_argument("--bQ", type=float, default=-6.0)
    parser.add_argument("--cQ", type=float, default=6.0)
    parser.add_argument("--wall-buffer", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_inputs(args)

    shape = tuple(args.shape)
    lengths = tuple(args.lengths)
    center = tuple(args.center)
    spacing = tuple(length / count for count, length in zip(shape, lengths))
    epsilon0 = math.radians(args.epsilon_deg)
    epsilon_decay_length = (
        args.epsilon_decay_length
        if args.epsilon_decay_length > 0.0
        else args.radius
    )
    if args.S_bulk is not None:
        if args.S_bulk <= 0.0:
            raise ValueError("S_bulk must be positive")
        S_bulk = args.S_bulk
    else:
        S_bulk = positive_equilibrium_S(args.aQ, args.bQ, args.cQ)
    radial_transition_size = (
        args.radial_transition_size
        if args.radial_transition_size > 0.0
        else args.core_size
    )

    fields = construct_loop(
        shape=shape,
        lengths=lengths,
        center=center,
        radius=args.radius,
        m_axis_name=args.m_axis,
        half_thickness=args.half_thickness,
        epsilon0=epsilon0,
        epsilon_decay_length=epsilon_decay_length,
        core_size=args.core_size,
        radial_transition_size=radial_transition_size,
        S_bulk=S_bulk,
        wall_buffer=args.wall_buffer,
    )
    validation = validation_summary(
        fields,
        S_bulk=S_bulk,
    )
    if not validation["finite_Q"]:
        raise RuntimeError("constructed Q contains non-finite values")

    loop_convention = convention_metadata(
        fields["N"],
        fields["M"],
        fields["L"],
        fields["N"],
        fields["N"],
        fields["L"],
    )
    if loop_convention["classified_type"] != "pure-splay":
        raise RuntimeError(f"Unexpected ideal-loop classification: {loop_convention}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "Q": args.output_dir / "Q_pure_splay_loop.npy",
        "director_model": args.output_dir / "n_pure_splay_loop.npy",
        "S_model": args.output_dir / "S_pure_splay_loop.npy",
        "director_defined_mask": args.output_dir / "director_defined_mask.npy",
        "inside_angle": args.output_dir / "theta_inside.npy",
        "outside_angle": args.output_dir / "theta_outside.npy",
        "inside_weight": args.output_dir / "inside_weight.npy",
        "wall_blend_weight": args.output_dir / "wall_blend_weight.npy",
        "metadata": args.output_dir / "metadata.json",
    }
    for name in (
        "Q",
        "director_model",
        "S_model",
        "director_defined_mask",
        "inside_angle",
        "outside_angle",
        "inside_weight",
        "wall_blend_weight",
    ):
        np.save(paths[name], fields[name])

    metadata = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "S_initial": S_bulk,
                "S_bulk": S_bulk,
                "aQ": args.aQ,
                "bQ": args.bQ,
                "cQ": args.cQ,
            },
        },
        "classification": {
            "type": "pure-splay",
            "classification_regime": "ideal-loop one-dimensional skyrmion",
            "gamma_rad": 0.5 * math.pi,
            "gamma_deg": 90.0,
            "sigma_rad": 0.0,
            "sigma_deg": 0.0,
            "N": fields["N"].tolist(),
            "M": fields["M"].tolist(),
            "L": fields["L"].tolist(),
            "k": fields["N"].tolist(),
            "nu": fields["N"].tolist(),
            "Omega": fields["L"].tolist(),
            "principal_plane": "N-M",
            "loop_plane": "M-L",
        },
        "loop_convention": loop_convention,
        "domain": {
            "shape": list(shape),
            "lengths": list(lengths),
            "spacing": list(spacing),
            "boundary_conditions_Q": ["periodic", "neumann", "neumann"],
        },
        "loop": {
            "center": list(center),
            "radius": args.radius,
            "half_thickness": args.half_thickness,
            "epsilon0_rad": epsilon0,
            "epsilon0_deg": args.epsilon_deg,
            "epsilon_decay_length": epsilon_decay_length,
            "core_size": args.core_size,
            "radial_transition_size": radial_transition_size,
            "wall_buffer": args.wall_buffer,
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "validation": validation,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    print(f"Saved pure-splay loop Q field: {paths['Q'].resolve()}")
    print(
        "Geometry: "
        f"center={center} radius={args.radius} h={args.half_thickness} "
        f"M={args.m_axis}"
    )
    print(
        "Material profile: "
        f"S_bulk={S_bulk:.9f} core={args.core_size} "
        f"epsilon={args.epsilon_deg:.3f} deg "
        f"epsilon_decay={epsilon_decay_length}"
    )
    print(
        "Validation: "
        f"finite={validation['finite_Q']} "
        f"min_S={validation['S_min']:.6f} "
        f"wall_delta_max={validation['max_adjacent_wall_cell_Q_difference']:.3e} "
        f"defect_faces={validation['defect_plaquette_counts']}"
    )


if __name__ == "__main__":
    main()
