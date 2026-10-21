"""Shared, allocation-free public complete-stress validation.

The public Periodic and Channel compilers consume the same complete-stress
equation declaration and the same compact-Q snapshot convention.  Keep the
pre-runtime checks here so the two application adapters cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path

import numpy as np


_RUNTIME_PARAMETER_KEYS = (
    "material",
    "ldg_l1",
    "activity_amplitude",
    "stokes",
)
_MATERIAL_PARAMETER_KEYS = (
    "ldg_a",
    "ldg_b",
    "ldg_c",
    "gamma",
    "flow_alignment",
    "beta",
)
_STOKES_PARAMETER_KEYS = (
    "viscosity",
    "friction",
    "pressure_gauge",
    "tangential_zero_mode_policy",
)


def _finite_real(
    value: object,
    description: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (non_negative and float(value) < 0.0)
    ):
        qualifier = (
            "positive and finite"
            if positive
            else "non-negative and finite"
            if non_negative
            else "finite"
        )
        raise ValueError(f"{description} must be a {qualifier} real number")
    return float(value)


def validate_complete_stress_runtime_parameters(
    parameters: Mapping[str, object],
) -> None:
    """Require every parameter consumed by either complete-stress runtime.

    Canonical public constructors already validate the values.  This check is
    a fail-closed boundary for declarations that callers have subsequently
    altered with tools such as :func:`dataclasses.replace`: no missing key may
    survive public compilation and leak as a runtime ``KeyError``.
    """

    if not isinstance(parameters, Mapping):
        raise TypeError("complete-stress model parameters must be a mapping")
    for name in _RUNTIME_PARAMETER_KEYS:
        parameters[name]

    material = parameters["material"]
    if not isinstance(material, Mapping):
        raise TypeError("complete-stress material must be a mapping")
    for name in _MATERIAL_PARAMETER_KEYS:
        _finite_real(
            material[name],
            f"complete-stress material {name}",
            positive=name == "gamma",
        )

    _finite_real(
        parameters["ldg_l1"],
        "complete-stress ldg_l1",
        positive=True,
    )
    _finite_real(
        parameters["activity_amplitude"],
        "complete-stress activity_amplitude",
    )

    stokes = parameters["stokes"]
    if not isinstance(stokes, Mapping):
        raise TypeError("complete-stress Stokes declaration must be a mapping")
    stokes_parameters = stokes["parameters"]
    if not isinstance(stokes_parameters, Mapping):
        raise TypeError(
            "complete-stress Stokes parameters must be a mapping"
        )
    for name in _STOKES_PARAMETER_KEYS:
        stokes_parameters[name]
    _finite_real(
        stokes_parameters["viscosity"],
        "complete-stress viscosity",
        positive=True,
    )
    _finite_real(
        stokes_parameters["friction"],
        "complete-stress friction",
        non_negative=True,
    )
    for name in ("pressure_gauge", "tangential_zero_mode_policy"):
        if not isinstance(stokes_parameters[name], str):
            raise TypeError(f"complete-stress {name} must be a string")


def validate_existing_q_snapshot_shape(
    snapshot: Mapping[str, object],
    grid_shape: tuple[int, ...],
    *,
    component_count: int,
) -> None:
    """Validate an existing compact-Q ``.npy`` shape without reading payload.

    Compilation remains usable for declaration/planning workflows whose
    external artifact has not been staged yet.  Once the declared file exists,
    however, its shape is part of the compilation contract.  Runtime loading
    deliberately repeats this check and additionally validates dtype and
    finite values, closing the time-of-check/time-of-use boundary.
    """

    path = Path(snapshot["directory"]).expanduser().resolve() / (
        f"Q_{snapshot['step']}.npy"
    )
    if not path.exists():
        return
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError("Q snapshot header could not be read") from exc
    try:
        observed = tuple(int(size) for size in values.shape)
    finally:
        memory_map = getattr(values, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    expected_component_last = (*grid_shape, component_count)
    expected_component_first = (component_count, *grid_shape)
    if observed not in {expected_component_last, expected_component_first}:
        raise ValueError(
            "Q snapshot shape does not match the declared geometry: "
            f"observed={observed!r}, expected one of "
            f"{expected_component_last!r} or {expected_component_first!r}"
        )


__all__ = [
    "validate_complete_stress_runtime_parameters",
    "validate_existing_q_snapshot_shape",
]
