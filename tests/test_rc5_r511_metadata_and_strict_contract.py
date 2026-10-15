"""R5-11 Periodic projection metadata and lifted fresh-state semantics."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from pssolver.functional import (
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
    build_periodic_functional_checkpoint_identity,
)
from pssolver.functional.checkpoint_identity import (
    normalize_functional_checkpoint_identity,
)
from pssolver.functional.legacy_identity import (
    legacy_runtime_identity_sha256,
)
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
)
from pssolver.operators.projection import (
    periodic_hermitian_state_projection,
)


def _record(metadata):
    return {
        "metadata": metadata,
        "sha256": hashlib.sha256(
            json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _identity(
    storage: str,
    projection: str,
) -> tuple[FunctionalRuntimeIdentity, FunctionalStateSpec]:
    physical_shape = (5, 1, 6, 6, 4)
    spectral_shape = (
        (5, 1, 6, 4, 4)
        if storage == "hermitian_half"
        else physical_shape
    )
    state_spec = FunctionalStateSpec(
        (
            FunctionalTensorSpec(
                name="q_physical",
                shape=physical_shape,
                dtype="float64",
                device="cpu",
                batch_axis=1,
                layout="component_batch_xyz",
                meaning="test physical Q",
            ),
            FunctionalTensorSpec(
                name="q_spectral",
                shape=spectral_shape,
                dtype="complex128",
                device="cpu",
                batch_axis=1,
                layout="component_batch_native_spectral_xyz",
                meaning="test spectral Q",
            ),
        )
    )
    execution = _record(
        {
            "backend": "torch_spectral",
            "runtime_path": "periodic_spectral",
            "options": {"tf32": "off"},
        }
    )
    execution["functional_runtime"] = {
        "kind": "periodic_activity_batch_one",
        "device": "cpu",
        "pointwise_execution": "eager",
        "q_gradient_reuse": False,
        "pressure_solver": "direct_periodic_fourier",
        "spectral_refresh": "disabled",
        "hermitian_state_projection": projection,
        "activity_product_before_divergence": True,
        "snapshot_sha256": "a" * 64,
        "fallback_allowed": False,
        "fallback_used": False,
        "deterministic_replay": "bitwise",
        "durable_checkpoint_bridge_version": 1,
    }
    return (
        FunctionalRuntimeIdentity(
            scientific=_record({"model": "complete_stress_beris_edwards"}),
            discretization=_record(
                {
                    "grid": {"shape": [6, 6, 4]},
                    "numerics": {"spectral_storage": storage},
                }
            ),
            execution=execution,
            state_layout=state_spec.to_metadata(),
        ),
        state_spec,
    )


@pytest.mark.parametrize(
    ("storage", "expected"),
    (
        (
            "full_complex",
            "complete_conjugate_pair_projection_each_step.v2",
        ),
        (
            "hermitian_half",
            "self_conjugate_stored_planes_projection_each_step.v2",
        ),
    ),
)
def test_periodic_projection_description_matches_checkpoint_identity(
    storage,
    expected,
):
    assert periodic_hermitian_state_projection(storage) == expected


def test_pre_r5_11_current_checkpoint_identity_normalizes_without_dynamics_change():
    storage = "full_complex"
    current_projection = periodic_hermitian_state_projection(storage)
    current_runtime, state_spec = _identity(storage, current_projection)
    current = build_periodic_functional_checkpoint_identity(
        functional_identity=current_runtime,
        state_spec=state_spec,
    )

    old = copy.deepcopy(current.to_metadata())
    old_execution = old["forward_dynamics"]["payload"]["execution"]
    old_execution["functional_runtime"]["hermitian_state_projection"] = (
        "self_conjugate_planes_each_step"
    )
    old["derivative_dynamics"]["payload"][
        "spectral_state_projection"
    ] = "self_conjugate_planes_each_step"
    old_identity = CheckpointCompatibilityIdentity.from_metadata(old)

    assert normalize_functional_checkpoint_identity(
        old_identity.to_metadata()
    ) == normalize_functional_checkpoint_identity(current.to_metadata())


@pytest.mark.parametrize("generation", ("rc1", "rc2", "rc3"))
def test_released_periodic_identity_literal_stays_frozen(generation):
    current, _state_spec = _identity(
        "hermitian_half",
        periodic_hermitian_state_projection("hermitian_half"),
    )
    current = current.to_metadata()
    historical = copy.deepcopy(current)
    historical["execution"]["functional_runtime"][
        "hermitian_state_projection"
    ] = "self_conjugate_planes_each_step"

    assert legacy_runtime_identity_sha256(
        current,
        family="periodic",
        source_release_generation=generation,
    ) == legacy_runtime_identity_sha256(
        historical,
        family="periodic",
        source_release_generation=generation,
    )


def test_unknown_periodic_storage_is_rejected():
    with pytest.raises(ValueError, match="unsupported periodic spectral storage"):
        periodic_hermitian_state_projection("unknown")
