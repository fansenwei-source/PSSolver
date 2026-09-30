"""Provisional, tensor-explicit contracts for functional PDE execution.

P9.1 defines declarations only.  This module does not construct a runtime,
allocate state, execute a timestep, or claim differentiability.  The API is
deliberately not re-exported from :mod:`pssolver` while it remains provisional.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import torch

from pssolver.configuration.simulation import SimulationSpec


FUNCTIONAL_API_VERSION = "0.1-provisional"
FUNCTIONAL_OBSERVATION_TIME = "input_state_under_current_control"

FunctionalState: TypeAlias = tuple[torch.Tensor, ...]
FunctionalControls: TypeAlias = Mapping[str, torch.Tensor]
FunctionalObservations: TypeAlias = Mapping[str, torch.Tensor]

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{description} must be a Python identifier")
    return value


def _nonempty(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FunctionalTensorSpec:
    """Exact tensor identity; validation never converts dtype or device."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    batch_axis: int
    layout: str
    meaning: str
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.name, "tensor name")
        shape = tuple(self.shape)
        if not shape or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shape
        ):
            raise ValueError("tensor shape must contain positive integers")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported functional tensor dtype: {self.dtype!r}")
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("functional tensor device is invalid") from exc
        if device.type == "cuda" and device.index is None:
            raise ValueError("CUDA functional tensor identity requires an index")
        if (
            not isinstance(self.batch_axis, int)
            or isinstance(self.batch_axis, bool)
            or not 0 <= self.batch_axis < len(shape)
        ):
            raise ValueError("batch_axis must index the tensor shape")
        _nonempty(self.layout, "tensor layout")
        _nonempty(self.meaning, "tensor meaning")
        components = tuple(self.component_names)
        if components:
            if any(
                not isinstance(value, str) or not value.isidentifier()
                for value in components
            ):
                raise ValueError("component_names must be Python identifiers")
            if len(set(components)) != len(components):
                raise ValueError("component_names must be unique")
            if len(components) not in shape:
                raise ValueError(
                    "one tensor axis must match the component_names length"
                )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "device", str(device))
        object.__setattr__(self, "component_names", components)

    @property
    def batch_size(self) -> int:
        return self.shape[self.batch_axis]

    def validate(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{self.name} must be a torch.Tensor")
        if tuple(value.shape) != self.shape:
            raise ValueError(
                f"{self.name} shape must be {self.shape}, got {tuple(value.shape)}"
            )
        if value.dtype is not _DTYPES[self.dtype]:
            raise TypeError(
                f"{self.name} dtype must be {self.dtype}, got {value.dtype}"
            )
        if str(value.device) != self.device:
            raise ValueError(
                f"{self.name} device must be {self.device}, got {value.device}"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "batch_axis": self.batch_axis,
            "layout": self.layout,
            "meaning": self.meaning,
            "component_names": list(self.component_names),
        }


@dataclass(frozen=True, slots=True)
class FunctionalStateSpec:
    """Ordered flat-tuple state containing every persistent next-step value."""

    components: tuple[FunctionalTensorSpec, ...]
    layout_version: int = 1

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or any(
            not isinstance(value, FunctionalTensorSpec) for value in components
        ):
            raise TypeError("state components must be FunctionalTensorSpec values")
        if len({value.name for value in components}) != len(components):
            raise ValueError("state component names must be unique")
        batch_sizes = {value.batch_size for value in components}
        if len(batch_sizes) != 1:
            raise ValueError("all state components must use one batch size")
        if (
            not isinstance(self.layout_version, int)
            or isinstance(self.layout_version, bool)
            or self.layout_version <= 0
        ):
            raise ValueError("layout_version must be a positive integer")
        object.__setattr__(self, "components", components)

    @property
    def batch_size(self) -> int:
        return self.components[0].batch_size

    def validate(self, state: FunctionalState) -> None:
        if not isinstance(state, tuple):
            raise TypeError("functional state must be a flat tuple")
        if len(state) != len(self.components):
            raise ValueError("functional state component count is incorrect")
        for specification, value in zip(self.components, state, strict=True):
            specification.validate(value)

    def to_metadata(self) -> dict[str, object]:
        return {
            "container": "flat_tuple",
            "layout_version": self.layout_version,
            "batch_size": self.batch_size,
            "components": [value.to_metadata() for value in self.components],
        }


@dataclass(frozen=True, slots=True)
class FunctionalControlFieldSpec:
    """Model-owned control-field meaning and injection contract."""

    name: str
    tensor: FunctionalTensorSpec
    equation_term: str
    injection_order: str
    dealiasing_identity: str
    grid_location: str
    broadcast_rules: tuple[str, ...]
    admissible_min: float | None = None
    admissible_max: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "control field name")
        if not isinstance(self.tensor, FunctionalTensorSpec):
            raise TypeError("control tensor must be a FunctionalTensorSpec")
        if self.tensor.name != self.name:
            raise ValueError("control field and tensor names must match")
        _nonempty(self.equation_term, "equation_term")
        _nonempty(self.injection_order, "injection_order")
        _nonempty(self.dealiasing_identity, "dealiasing_identity")
        _nonempty(self.grid_location, "grid_location")
        broadcast_rules = tuple(self.broadcast_rules)
        if not broadcast_rules or any(
            not isinstance(value, str) or not value.strip()
            for value in broadcast_rules
        ):
            raise ValueError("broadcast_rules must contain explicit rules")
        object.__setattr__(self, "broadcast_rules", broadcast_rules)
        lower, upper = self.admissible_min, self.admissible_max
        if any(
            value is not None
            and (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            )
            for value in (lower, upper)
        ):
            raise ValueError("admissible bounds must be finite real values")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("admissible_min cannot exceed admissible_max")

    def validate(self, value: torch.Tensor) -> None:
        self.tensor.validate(value)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"control field {self.name} contains NaN or Inf")
        if self.admissible_min is not None and bool(
            (value < self.admissible_min).any()
        ):
            raise ValueError(f"control field {self.name} is below its lower bound")
        if self.admissible_max is not None and bool(
            (value > self.admissible_max).any()
        ):
            raise ValueError(f"control field {self.name} is above its upper bound")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tensor": self.tensor.to_metadata(),
            "equation_term": self.equation_term,
            "injection_order": self.injection_order,
            "dealiasing_identity": self.dealiasing_identity,
            "grid_location": self.grid_location,
            "broadcast_rules": list(self.broadcast_rules),
            "admissible_bounds": [self.admissible_min, self.admissible_max],
        }


@dataclass(frozen=True, slots=True)
class FunctionalObservationSpec:
    """Named tensor observation with an explicit time convention."""

    name: str
    tensor: FunctionalTensorSpec
    convention: str
    control_dependent: bool
    terminal_available_without_control: bool
    time_alignment: str = FUNCTIONAL_OBSERVATION_TIME

    def __post_init__(self) -> None:
        _identifier(self.name, "observation name")
        if not isinstance(self.tensor, FunctionalTensorSpec):
            raise TypeError("observation tensor must be a FunctionalTensorSpec")
        if self.tensor.name != self.name:
            raise ValueError("observation and tensor names must match")
        _nonempty(self.convention, "observation convention")
        if self.time_alignment != FUNCTIONAL_OBSERVATION_TIME:
            raise ValueError("unsupported functional observation time alignment")
        if not isinstance(self.control_dependent, bool) or not isinstance(
            self.terminal_available_without_control,
            bool,
        ):
            raise TypeError("observation dependency flags must be bool values")
        if self.control_dependent and self.terminal_available_without_control:
            raise ValueError(
                "a control-dependent observation cannot be a terminal "
                "observation without control"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tensor": self.tensor.to_metadata(),
            "convention": self.convention,
            "control_dependent": self.control_dependent,
            "terminal_available_without_control": (
                self.terminal_available_without_control
            ),
            "time_alignment": self.time_alignment,
        }


@dataclass(frozen=True, slots=True)
class FunctionalCapabilitySet:
    """Fail-closed claims made by one concrete functional runtime."""

    supported_batch_sizes: tuple[int, ...] = (1,)
    pure_step: bool = False
    combined_step_and_observe: bool = False
    deterministic_replay: str = "not_qualified"
    differentiability: str = "not_qualified"
    durable_checkpoint_bridge: bool = False
    explicit_jvp: bool = False
    explicit_vjp: bool = False
    inner_solve_gradient: str = "not_applicable"
    differentiable_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sizes = tuple(self.supported_batch_sizes)
        if not sizes or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in sizes
        ):
            raise ValueError("supported_batch_sizes must be positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError("supported_batch_sizes must be unique")
        if self.deterministic_replay not in {"not_qualified", "bitwise"}:
            raise ValueError("unsupported deterministic replay capability")
        if self.differentiability not in {
            "not_qualified",
            "torch_autograd",
            "validated_custom_adjoint",
        }:
            raise ValueError("unsupported differentiability capability")
        flags = (
            self.pure_step,
            self.combined_step_and_observe,
            self.durable_checkpoint_bridge,
            self.explicit_jvp,
            self.explicit_vjp,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("functional capability flags must be bool values")
        _nonempty(self.inner_solve_gradient, "inner_solve_gradient")
        differentiable_inputs = tuple(self.differentiable_inputs)
        if any(
            not isinstance(value, str) or not value.isidentifier()
            for value in differentiable_inputs
        ):
            raise ValueError("differentiable_inputs must be Python identifiers")
        if len(set(differentiable_inputs)) != len(differentiable_inputs):
            raise ValueError("differentiable_inputs must be unique")
        if self.differentiability == "not_qualified" and differentiable_inputs:
            raise ValueError(
                "unqualified differentiability cannot claim differentiable inputs"
            )
        object.__setattr__(self, "supported_batch_sizes", sizes)
        object.__setattr__(self, "differentiable_inputs", differentiable_inputs)

    def to_metadata(self) -> dict[str, object]:
        return {
            "supported_batch_sizes": list(self.supported_batch_sizes),
            "pure_step": self.pure_step,
            "combined_step_and_observe": self.combined_step_and_observe,
            "deterministic_replay": self.deterministic_replay,
            "differentiability": self.differentiability,
            "durable_checkpoint_bridge": self.durable_checkpoint_bridge,
            "explicit_jvp": self.explicit_jvp,
            "explicit_vjp": self.explicit_vjp,
            "inner_solve_gradient": self.inner_solve_gradient,
            "differentiable_inputs": list(self.differentiable_inputs),
        }


@dataclass(frozen=True, slots=True)
class FunctionalRuntimeIdentity:
    """Immutable JSON identity for scientific, discretization, and execution facts."""

    scientific: Mapping[str, object]
    discretization: Mapping[str, object]
    execution: Mapping[str, object]
    state_layout: Mapping[str, object]
    api_version: str = FUNCTIONAL_API_VERSION
    _canonical_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.api_version != FUNCTIONAL_API_VERSION:
            raise ValueError("unsupported provisional functional API version")
        payload = {
            "api_version": self.api_version,
            "scientific": dict(self.scientific),
            "discretization": dict(self.discretization),
            "execution": dict(self.execution),
            "state_layout": dict(self.state_layout),
        }
        try:
            canonical = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("functional runtime identity must be JSON-safe") from exc
        frozen = json.loads(canonical)
        for name in ("scientific", "discretization", "execution", "state_layout"):
            object.__setattr__(self, name, _freeze_json(frozen[name]))
        object.__setattr__(self, "_canonical_json", canonical)

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True, slots=True)
class FunctionalCheckpointState:
    """Owned state returned by one durable checkpoint import."""

    state: FunctionalState
    completed_steps: int
    source_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, tuple) or any(
            not isinstance(value, torch.Tensor) for value in self.state
        ):
            raise TypeError("checkpoint state must be a flat tensor tuple")
        if (
            not isinstance(self.completed_steps, int)
            or isinstance(self.completed_steps, bool)
            or self.completed_steps < 0
        ):
            raise ValueError("completed_steps must be non-negative")
        _nonempty(self.source_format, "source_format")


@runtime_checkable
class FunctionalCheckpointBridgeProtocol(Protocol):
    """Versioned conversion between functional and production checkpoint state."""

    @property
    def format_version(self) -> int: ...

    def export_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
    ) -> Path: ...

    def import_checkpoint(
        self,
        directory: str | Path,
    ) -> FunctionalCheckpointState: ...


@dataclass(frozen=True, slots=True)
class FunctionalRuntimeConstructionRequest:
    """Explicit request consumed by a future functional-runtime factory."""

    simulation: SimulationSpec
    control_fields: tuple[FunctionalControlFieldSpec, ...]
    observations: tuple[FunctionalObservationSpec, ...]
    batch_size: int = 1
    api_version: str = FUNCTIONAL_API_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        controls = tuple(self.control_fields)
        observations = tuple(self.observations)
        if not controls or any(
            not isinstance(value, FunctionalControlFieldSpec)
            for value in controls
        ):
            raise TypeError("control_fields must contain functional controls")
        if not observations or any(
            not isinstance(value, FunctionalObservationSpec)
            for value in observations
        ):
            raise TypeError("observations must contain functional observations")
        if len({value.name for value in controls}) != len(controls):
            raise ValueError("functional control names must be unique")
        if len({value.name for value in observations}) != len(observations):
            raise ValueError("functional observation names must be unique")
        if self.batch_size != 1:
            raise ValueError("P9.1 construction declarations require batch_size=1")
        if self.api_version != FUNCTIONAL_API_VERSION:
            raise ValueError("unsupported provisional functional API version")
        if any(value.tensor.batch_size != self.batch_size for value in controls):
            raise ValueError("control tensor batch size does not match the request")
        if any(
            value.tensor.batch_size != self.batch_size for value in observations
        ):
            raise ValueError(
                "observation tensor batch size does not match the request"
            )
        object.__setattr__(self, "control_fields", controls)
        object.__setattr__(self, "observations", observations)

    def to_metadata(self) -> dict[str, object]:
        identities = self.simulation.identity_metadata()
        return {
            "api_version": self.api_version,
            "batch_size": self.batch_size,
            "simulation_sha256": self.simulation.canonical_sha256(),
            "simulation_identity": {
                name: identities[name]
                for name in ("scientific", "discretization", "execution")
            },
            "control_fields": [value.to_metadata() for value in self.control_fields],
            "observations": [value.to_metadata() for value in self.observations],
        }


@dataclass(frozen=True, slots=True)
class FunctionalRuntimeDeclaration:
    """Non-executable declaration of one future functional runtime.

    The declaration binds the prospective state layout and runtime identity to
    a canonical construction request without claiming that a factory, step,
    observation, checkpoint bridge, or derivative is available.
    """

    request: FunctionalRuntimeConstructionRequest
    state_spec: FunctionalStateSpec
    capabilities: FunctionalCapabilitySet
    identity: FunctionalRuntimeIdentity
    executable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request, FunctionalRuntimeConstructionRequest):
            raise TypeError("request must be a functional construction request")
        if not isinstance(self.state_spec, FunctionalStateSpec):
            raise TypeError("state_spec must be a FunctionalStateSpec")
        if not isinstance(self.capabilities, FunctionalCapabilitySet):
            raise TypeError("capabilities must be a FunctionalCapabilitySet")
        if not isinstance(self.identity, FunctionalRuntimeIdentity):
            raise TypeError("identity must be a FunctionalRuntimeIdentity")
        if not isinstance(self.executable, bool):
            raise TypeError("executable must be a bool")
        if self.request.batch_size != self.state_spec.batch_size:
            raise ValueError("request and state batch sizes differ")
        if self.request.batch_size not in self.capabilities.supported_batch_sizes:
            raise ValueError("capabilities do not declare the request batch size")
        if self.request.api_version != self.identity.api_version:
            raise ValueError("request and identity API versions differ")
        if (
            self.identity.to_metadata()["state_layout"]
            != self.state_spec.to_metadata()
        ):
            raise ValueError("identity state layout differs from the declaration")
        if not self.executable and (
            self.capabilities.pure_step
            or self.capabilities.combined_step_and_observe
            or self.capabilities.deterministic_replay != "not_qualified"
            or self.capabilities.differentiability != "not_qualified"
            or self.capabilities.durable_checkpoint_bridge
            or self.capabilities.explicit_jvp
            or self.capabilities.explicit_vjp
            or (
                self.capabilities.inner_solve_gradient
                not in {"not_applicable", "not_qualified"}
            )
            or self.capabilities.differentiable_inputs
        ):
            raise ValueError(
                "a non-executable declaration cannot claim runtime capabilities"
            )

    @property
    def control_specs(self) -> tuple[FunctionalControlFieldSpec, ...]:
        return self.request.control_fields

    @property
    def observation_specs(self) -> tuple[FunctionalObservationSpec, ...]:
        return self.request.observations

    def to_metadata(self) -> dict[str, object]:
        return {
            "executable": self.executable,
            "request": self.request.to_metadata(),
            "state_spec": self.state_spec.to_metadata(),
            "capabilities": self.capabilities.to_metadata(),
            "identity": self.identity.to_metadata(),
            "identity_sha256": self.identity.canonical_sha256(),
        }


@runtime_checkable
class FunctionalRuntimeProtocol(Protocol):
    """Provisional protocol; no implementation is provided by P9.1."""

    @property
    def api_version(self) -> str: ...

    @property
    def state_spec(self) -> FunctionalStateSpec: ...

    @property
    def control_specs(self) -> tuple[FunctionalControlFieldSpec, ...]: ...

    @property
    def control_field_schema(self) -> tuple[FunctionalControlFieldSpec, ...]: ...

    @property
    def observation_specs(self) -> tuple[FunctionalObservationSpec, ...]: ...

    @property
    def capabilities(self) -> FunctionalCapabilitySet: ...

    @property
    def checkpoint_bridge(self) -> FunctionalCheckpointBridgeProtocol | None: ...

    def identity(self) -> FunctionalRuntimeIdentity: ...

    def initial_state(self) -> FunctionalState: ...

    def step(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> FunctionalState: ...

    def observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls | None,
    ) -> FunctionalObservations: ...

    def step_and_observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> tuple[FunctionalState, FunctionalObservations]: ...


@runtime_checkable
class FunctionalRuntimeFactoryProtocol(Protocol):
    """Explicit construction boundary for a future functional runtime."""

    def build(
        self,
        request: FunctionalRuntimeConstructionRequest,
    ) -> FunctionalRuntimeProtocol: ...


__all__ = [
    "FUNCTIONAL_API_VERSION",
    "FUNCTIONAL_OBSERVATION_TIME",
    "FunctionalCapabilities",
    "FunctionalCapabilitySet",
    "FunctionalCheckpointBridgeProtocol",
    "FunctionalCheckpointState",
    "FunctionalControlFieldSpec",
    "FunctionalControls",
    "FunctionalObservationSpec",
    "FunctionalObservations",
    "FunctionalRuntimeConstructionRequest",
    "FunctionalRuntimeDeclaration",
    "FunctionalRuntimeFactoryProtocol",
    "FunctionalRuntimeIdentity",
    "FunctionalRuntimeProtocol",
    "FunctionalState",
    "FunctionalStateSpec",
    "FunctionalTensorSpec",
]

# Readable alias retained within the provisional module only.
FunctionalCapabilities = FunctionalCapabilitySet
