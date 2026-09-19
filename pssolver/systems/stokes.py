"""Backend-independent contracts for incompressible Stokes systems."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .algebraic import AlgebraicSystemSpec


INCOMPRESSIBLE_STOKES_CAPABILITY = "incompressible_stokes"


class PressureGauge(str, Enum):
    """Supported pressure reference choices."""

    ZERO_MEAN = "zero_mean"


class TangentialZeroModePolicy(str, Enum):
    """Treatment of geometry-dependent uniform tangential velocity modes."""

    ZERO_MEAN = "zero_mean"
    FRICTION = "friction"
    NOT_APPLICABLE = "not_applicable"


def _component_tuple(
    values: tuple[str, ...],
    description: str,
) -> tuple[str, str, str]:
    if isinstance(values, str):
        raise TypeError(f"{description} must be an iterable, not a string")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{description} must be iterable") from exc
    if len(values) != 3 or any(
        not isinstance(name, str) or not name.isidentifier()
        for name in values
    ):
        raise ValueError(
            f"{description} must contain three Python identifiers"
        )
    if len(set(values)) != 3:
        raise ValueError(f"{description} must contain unique names")
    return values


def _finite_coefficient(
    value: object,
    description: str,
    *,
    positive: bool,
) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be finite and {qualifier}")
    return float(value)


@dataclass(frozen=True, slots=True)
class IncompressibleStokesSystemSpec:
    """Physical/run choices for ``-grad(p)+eta*Laplacian(u)-f*u+F=0``.

    Component order is coordinate order.  Component-specific boundary spaces
    remain attached to the corresponding :class:`FieldSpec`; this request
    names their roles without selecting a transform or concrete solver.
    """

    name: str
    force_components: tuple[str, str, str]
    velocity_components: tuple[str, str, str]
    pressure_component: str
    viscosity: float
    friction: float = 0.0
    pressure_gauge: PressureGauge = PressureGauge.ZERO_MEAN
    tangential_zero_mode_policy: TangentialZeroModePolicy = (
        TangentialZeroModePolicy.ZERO_MEAN
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Stokes system name must be a Python identifier")
        forces = _component_tuple(self.force_components, "force_components")
        velocities = _component_tuple(
            self.velocity_components,
            "velocity_components",
        )
        if (
            not isinstance(self.pressure_component, str)
            or not self.pressure_component.isidentifier()
        ):
            raise ValueError(
                "pressure_component must be a Python identifier"
            )
        all_names = (*forces, *velocities, self.pressure_component)
        if len(set(all_names)) != len(all_names):
            raise ValueError("Stokes input and output component names must differ")
        viscosity = _finite_coefficient(
            self.viscosity,
            "viscosity",
            positive=True,
        )
        friction = _finite_coefficient(
            self.friction,
            "friction",
            positive=False,
        )
        if not isinstance(self.pressure_gauge, PressureGauge):
            raise TypeError("pressure_gauge must be a PressureGauge")
        policy = self.tangential_zero_mode_policy
        if not isinstance(policy, TangentialZeroModePolicy):
            raise TypeError(
                "tangential_zero_mode_policy must be a "
                "TangentialZeroModePolicy"
            )
        if policy is TangentialZeroModePolicy.ZERO_MEAN and friction != 0.0:
            raise ValueError("zero_mean tangential policy requires friction == 0")
        if policy is TangentialZeroModePolicy.FRICTION and friction <= 0.0:
            raise ValueError("friction tangential policy requires friction > 0")
        object.__setattr__(self, "force_components", forces)
        object.__setattr__(self, "velocity_components", velocities)
        object.__setattr__(self, "viscosity", viscosity)
        object.__setattr__(self, "friction", friction)

    def to_algebraic_system_spec(self) -> AlgebraicSystemSpec:
        """Lower the typed Stokes request to the generic dispatch contract."""

        return AlgebraicSystemSpec(
            name=self.name,
            capability=INCOMPRESSIBLE_STOKES_CAPABILITY,
            output_components=(
                *self.velocity_components,
                self.pressure_component,
            ),
            dependencies=self.force_components,
            parameters={
                "friction": self.friction,
                "pressure_gauge": self.pressure_gauge.value,
                "tangential_zero_mode_policy": (
                    self.tangential_zero_mode_policy.value
                ),
                "viscosity": self.viscosity,
            },
        )

    @classmethod
    def from_algebraic_system_spec(
        cls,
        system: AlgebraicSystemSpec,
    ) -> "IncompressibleStokesSystemSpec":
        """Validate and recover a typed request at a solver factory boundary."""

        if not isinstance(system, AlgebraicSystemSpec):
            raise TypeError("system must be an AlgebraicSystemSpec")
        if system.capability != INCOMPRESSIBLE_STOKES_CAPABILITY:
            raise ValueError(
                "algebraic system does not request incompressible Stokes"
            )
        if len(system.dependencies) != 3 or len(system.output_components) != 4:
            raise ValueError(
                "incompressible Stokes requires three forces and u_x/u_y/u_z/p"
            )
        expected = {
            "friction",
            "pressure_gauge",
            "tangential_zero_mode_policy",
            "viscosity",
        }
        if set(system.parameters) != expected:
            raise ValueError(
                "incompressible Stokes parameters must be exactly "
                f"{tuple(sorted(expected))!r}"
            )
        try:
            pressure_gauge = PressureGauge(system.parameters["pressure_gauge"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported pressure gauge") from exc
        try:
            zero_mode = TangentialZeroModePolicy(
                system.parameters["tangential_zero_mode_policy"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported tangential zero-mode policy") from exc
        return cls(
            name=system.name,
            force_components=system.dependencies,
            velocity_components=system.output_components[:3],
            pressure_component=system.output_components[3],
            viscosity=system.parameters["viscosity"],
            friction=system.parameters["friction"],
            pressure_gauge=pressure_gauge,
            tangential_zero_mode_policy=zero_mode,
        )

    def to_metadata(self) -> dict[str, object]:
        return self.to_algebraic_system_spec().to_metadata()
