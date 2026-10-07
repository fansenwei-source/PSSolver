"""Layered restart identities for stable functional checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import json

from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    CheckpointIdentityLayer,
    IdentityLayerKind,
)

from .contracts import FunctionalRuntimeIdentity, FunctionalStateSpec


_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint_export",
        "device",
        "durable_checkpoint_bridge_version",
        "fallback_allowed",
        "fallback_used",
        "production_checkpoint_import",
        "snapshot_sha256",
        "stage",
    }
)


def _owned_json(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _without_keys(value: object, excluded: frozenset[str]) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_keys(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, (tuple, list)):
        return [_without_keys(item, excluded) for item in value]
    return value


def _functional_runtime(
    identity: FunctionalRuntimeIdentity,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(identity, FunctionalRuntimeIdentity):
        raise TypeError("functional_identity must be a FunctionalRuntimeIdentity")
    metadata = identity.to_metadata()
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("functional identity execution metadata is missing")
    runtime = execution.get("functional_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("functional runtime identity metadata is missing")
    return metadata, dict(runtime)


def _forward_payload(
    identity: FunctionalRuntimeIdentity,
    *,
    family: CheckpointFamily,
) -> dict[str, object]:
    metadata, runtime = _functional_runtime(identity)
    pressure = runtime.get("pressure_solver")
    if family is CheckpointFamily.CHANNEL_FUNCTIONAL:
        if not isinstance(pressure, Mapping):
            raise ValueError("Channel functional pressure identity is missing")
        pressure = dict(pressure)
        pressure.pop("gradient", None)
        runtime["pressure_solver"] = pressure
    execution = dict(metadata["execution"])
    execution["functional_runtime"] = runtime
    return {
        "scientific": _owned_json(metadata["scientific"]),
        "discretization": _owned_json(metadata["discretization"]),
        "execution": _without_keys(execution, _PROVENANCE_KEYS),
    }


def _state_layout_payload(state_spec: FunctionalStateSpec) -> dict[str, object]:
    if not isinstance(state_spec, FunctionalStateSpec):
        raise TypeError("state_spec must be a FunctionalStateSpec")
    return {
        "components": _without_keys(
            state_spec.to_metadata(),
            frozenset({"device"}),
        ),
        "device_excluded": True,
        "ordering": "flat_tensor_tuple",
    }


def build_periodic_functional_checkpoint_identity(
    *,
    functional_identity: FunctionalRuntimeIdentity,
    state_spec: FunctionalStateSpec,
) -> CheckpointCompatibilityIdentity:
    """Build the current Periodic forward and autograd restart identity."""

    _metadata, runtime = _functional_runtime(functional_identity)
    if runtime.get("kind") != "periodic_activity_batch_one":
        raise ValueError("Periodic functional runtime kind is invalid")
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PERIODIC_FUNCTIONAL,
        runtime_path="periodic_activity_batch_one",
        forward_dynamics=CheckpointIdentityLayer(
            kind=IdentityLayerKind.FORWARD_DYNAMICS,
            version="periodic.functional.forward.v3",
            payload=_forward_payload(
                functional_identity,
                family=CheckpointFamily.PERIODIC_FUNCTIONAL,
            ),
        ),
        derivative_dynamics=CheckpointIdentityLayer(
            kind=IdentityLayerKind.DERIVATIVE_DYNAMICS,
            version="periodic.functional.derivative.v3",
            payload={
                "differentiability": "torch_autograd",
                "higher_order_derivatives": True,
                "inner_solve_gradient": "direct_periodic_fourier_autograd",
                "spectral_state_projection": runtime.get(
                    "hermitian_state_projection"
                ),
            },
        ),
        state_layout=CheckpointIdentityLayer(
            kind=IdentityLayerKind.STATE_LAYOUT,
            version="periodic.functional.state.v2",
            payload=_state_layout_payload(state_spec),
        ),
        backend_restart=CheckpointIdentityLayer(
            kind=IdentityLayerKind.BACKEND_RESTART,
            version="periodic.functional.backend.v2",
            payload={
                "persistent_hidden_state": [],
                "progress": "completed_steps",
                "reconstructed_caches": ["q_gradient"],
            },
        ),
    )


def build_channel_functional_checkpoint_identity(
    *,
    functional_identity: FunctionalRuntimeIdentity,
    state_spec: FunctionalStateSpec,
) -> CheckpointCompatibilityIdentity:
    """Build the current Channel forward and implicit-adjoint identity."""

    _metadata, runtime = _functional_runtime(functional_identity)
    if runtime.get("kind") != "channel_activity_batch_one":
        raise ValueError("Channel functional runtime kind is invalid")
    pressure = runtime.get("pressure_solver")
    if not isinstance(pressure, Mapping):
        raise ValueError("Channel functional pressure identity is missing")
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_FUNCTIONAL,
        runtime_path="channel_activity_batch_one",
        forward_dynamics=CheckpointIdentityLayer(
            kind=IdentityLayerKind.FORWARD_DYNAMICS,
            version="channel.functional.forward.v3",
            payload=_forward_payload(
                functional_identity,
                family=CheckpointFamily.CHANNEL_FUNCTIONAL,
            ),
        ),
        derivative_dynamics=CheckpointIdentityLayer(
            kind=IdentityLayerKind.DERIVATIVE_DYNAMICS,
            version="channel.functional.derivative.v3",
            payload={
                "gradient": pressure.get("gradient"),
                "higher_order_derivatives": False,
                "implicit_adjoint": "channel_pressure_implicit_adjoint.v1",
                "pressure_transpose_nyquist": (
                    "project_even_periodic_plane.v2"
                ),
                "transpose_action": (
                    "explicit_reverse_dataflow_conjugate_transpose"
                ),
                "transpose_initial_guess": "zero_every_call",
            },
        ),
        state_layout=CheckpointIdentityLayer(
            kind=IdentityLayerKind.STATE_LAYOUT,
            version="channel.functional.state.v2",
            payload=_state_layout_payload(state_spec),
        ),
        backend_restart=CheckpointIdentityLayer(
            kind=IdentityLayerKind.BACKEND_RESTART,
            version="channel.functional.backend.v2",
            payload={
                "persistent_hidden_state": [],
                "pressure_warm_start": "absent_functional_zero_start",
                "progress": "completed_steps",
                "reconstructed_caches": ["q_gradient"],
            },
        ),
    )


__all__ = [
    "build_channel_functional_checkpoint_identity",
    "build_periodic_functional_checkpoint_identity",
]
