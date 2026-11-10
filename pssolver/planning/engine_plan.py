"""Torch-free immutable execution plans for the v2 engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Literal

from pssolver.core._canonical import canonical_sha256


ENGINE_PLAN_SCHEMA_VERSION = 1


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _metadata(value):
    if is_dataclass(value):
        return {field.name: _metadata(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_metadata(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class GeometryPlan:
    kind: str
    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    axis_kinds: tuple[str, ...]
    spectral_storage: str
    hermitian_axis: int | None
    hermitian_state_projection: str
    transform_execution_order: str
    periodic_transform_execution: str


@dataclass(frozen=True, slots=True)
class FieldPlan:
    name: str
    role: Literal["evolved", "constrained"]
    components: tuple[str, ...]
    boundary_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransformGroup:
    components: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    indexing: Literal["contiguous_slice"]


@dataclass(frozen=True, slots=True)
class LinearTermPlan:
    target: str
    tier: Literal["diagonal"]
    recipe: str
    arguments: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "arguments", _freeze(self.arguments))


@dataclass(frozen=True, slots=True)
class ExplicitTermPlan:
    target: str
    kernel: str  # Package reference or model-local kernel identity.
    arguments: Mapping[str, object]
    pointwise_execution: Literal["eager", "compile"]
    q_gradient_reuse: bool

    def __post_init__(self):
        object.__setattr__(self, "arguments", _freeze(self.arguments))


@dataclass(frozen=True, slots=True)
class ConstraintPlan:
    key: tuple[str, str]
    solver: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arguments: Mapping[str, object]
    warm_start: Literal["stateless", "previous", "zero"]
    adjoint_mode: Literal["autograd", "custom_transpose"]
    pressure_diagnostics: bool

    def __post_init__(self):
        object.__setattr__(self, "arguments", _freeze(self.arguments))


@dataclass(frozen=True, slots=True)
class DealiasPlan:
    rule: str
    transform_execution: Literal["full", "truncated"]


@dataclass(frozen=True, slots=True)
class IntegratorPlan:
    scheme: Literal["semi_implicit_euler"]
    dt: float
    history_length: int
    refresh_interval: int | None


@dataclass(frozen=True, slots=True)
class ParameterPlan:
    name: str
    shape_kind: Literal["scalar", "field"]
    controllable: bool
    default: float | None


@dataclass(frozen=True, slots=True)
class OutputPlan:
    observables: tuple[str, ...]
    observation_point: Literal["after_constraints"]
    capture: str | None


@dataclass(frozen=True, slots=True)
class Plan:
    schema_version: int
    geometry: GeometryPlan
    fields: tuple[FieldPlan, ...]
    groups: tuple[TransformGroup, ...]
    linear: tuple[LinearTermPlan, ...]
    explicit: tuple[ExplicitTermPlan, ...]
    constraints: tuple[ConstraintPlan, ...]
    dealias: DealiasPlan
    integrator: IntegratorPlan
    parameters: tuple[ParameterPlan, ...]
    boundary_data: tuple[()]
    outputs: OutputPlan
    identity_sha256: str

    def to_metadata(self) -> dict[str, object]:
        return _metadata(self)


def build_plan(**members) -> Plan:
    """Bind the full plan once, including its implementation references."""
    if members.get("schema_version") != ENGINE_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported engine plan schema")
    if members.get("boundary_data") != ():
        raise ValueError("engine v2 requires homogeneous boundary data")
    members = {key: _freeze(value) for key, value in members.items()}
    return Plan(**members, identity_sha256=canonical_sha256(_metadata(members)))
