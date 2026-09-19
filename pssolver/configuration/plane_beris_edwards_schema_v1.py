"""Explicit compatibility serializer for the Plane schema-v1 contract.

This module is the sole implementation authority for the persistent metadata
and identity documents exposed by ``PlaneBerisEdwardsRunSpec``.  It is an
internal, direct-module-only adapter: the supported flat facade retains its
existing methods and delegates them here without changing their dispatch
semantics.

The serializer deliberately reads the facade's resolved public views.  It
does not serialize the provisional Phase 2 component graph, normalize paths,
or reinterpret any identity.  Those constraints preserve existing JSON bytes,
hashes, dry-run metadata, and format-v1 checkpoint compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from .plane_beris_edwards_declarations import DEFAULT_PLANE_RUNTIME_PATH

if TYPE_CHECKING:
    from .plane_beris_edwards import PlaneBerisEdwardsRunSpec


PLANE_RUN_SPEC_SCHEMA_VERSION = 1


class PlaneBerisEdwardsSchemaV1CompatibilitySerializer:
    """Stateless authority for the historical Plane schema-v1 documents."""

    __slots__ = ()

    def to_metadata(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> dict[str, object]:
        """Return the complete resolved, JSON-compatible run specification."""

        return {
            "schema_version": PLANE_RUN_SPEC_SCHEMA_VERSION,
            "authority": (
                "pssolver.configuration.PlaneBerisEdwardsRunSpec"
            ),
            "runtime_path": run_spec.runtime_path.value,
            "activity_number": run_spec.activity_number,
            "output_dir": str(run_spec.output_dir),
            "parameterization": run_spec.parameterization,
            "geometry": run_spec.geometry.to_metadata(),
            "boundaries": run_spec.boundaries.to_metadata(),
            "numerics": run_spec.numerics.to_metadata(),
            "model": {
                "ldg_a": run_spec.ldg_a,
                "ldg_b": run_spec.ldg_b,
                "ldg_c": run_spec.ldg_c,
                "gamma": run_spec.gamma,
                "flow_alignment": run_spec.flow_alignment,
                "eta": run_spec.eta,
                "beta": run_spec.beta,
                "initial_s": run_spec.initial_s,
                "molecular_field_linear_space": (
                    run_spec.molecular_field_linear_space
                ),
                "stress_divergence_sum_space": (
                    run_spec.stress_divergence_sum_space
                ),
                "pointwise_execution": run_spec.pointwise_execution,
            },
            "preset": run_spec.shendruk_preset.to_metadata(),
            "workflow": {
                "dt": run_spec.dt,
                "steps": run_spec.steps,
                "save_start_step": run_spec.save_start_step,
                "save_interval": run_spec.save_interval,
                "diagnostic_interval": run_spec.diagnostic_interval,
                "diagnostics": run_spec.diagnostics,
                "save_hydrodynamics": run_spec.save_hydrodynamics,
                "checkpoint_interval": run_spec.checkpoint_interval,
                "restart_from": (
                    str(run_spec.restart_from)
                    if run_spec.restart_from is not None
                    else None
                ),
                "spectral_refresh": run_spec.spectral_refresh.to_metadata(),
            },
            "initial_condition": {
                "seed": run_spec.seed,
                "num_defect_pairs": run_spec.num_defect_pairs,
                "defect_min_separation": run_spec.defect_min_separation,
                "defect_core_radius": run_spec.defect_core_radius,
                "background_angle": run_spec.background_angle,
                "twist_amplitude": run_spec.twist_amplitude,
                "twist_modes": list(run_spec.twist_modes),
            },
            "runtime": {
                "device": run_spec.device,
                "tf32": run_spec.tf32,
                "disable_q_gradient_reuse": (
                    run_spec.disable_q_gradient_reuse
                ),
                "validation_config_sha256": (
                    run_spec.validation_config_sha256
                ),
                "dry_run": run_spec.dry_run,
            },
            "zero_mode": {
                "policy": run_spec.zero_mode_policy,
                "friction_mode_fric": run_spec.friction_mode_fric,
            },
        }

    def canonical_sha256(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> str:
        """Hash the facade document through its supported public method."""

        return _canonical_sha256(run_spec.to_metadata())

    def runtime_identity_metadata(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> dict[str, object]:
        """Return the numerical identity required for same-backend restart."""

        metadata = run_spec.to_metadata()
        model = dict(metadata["model"])
        # The checkpoint supplies the evolved Q state, so the fresh-run
        # initializer amplitude is provenance rather than restart identity.
        model.pop("initial_s")
        return {
            "schema_version": PLANE_RUN_SPEC_SCHEMA_VERSION,
            "runtime_path": run_spec.runtime_path.value,
            "geometry": metadata["geometry"],
            "boundaries": metadata["boundaries"],
            "numerics": metadata["numerics"],
            "model": model,
            "preset": metadata["preset"],
            "dt": run_spec.dt,
            "spectral_refresh": run_spec.spectral_refresh.to_metadata(),
            "zero_mode": metadata["zero_mode"],
            "runtime_controls": {
                "device": run_spec.device,
                "tf32": run_spec.tf32,
                "disable_q_gradient_reuse": (
                    run_spec.disable_q_gradient_reuse
                ),
            },
        }

    def runtime_identity_sha256(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> str:
        """Hash restart identity through the facade's supported method."""

        return _canonical_sha256(run_spec.runtime_identity_metadata())

    def identity_metadata(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> dict[str, object]:
        """Return the historical compact configuration identity document."""

        return {
            "schema_version": PLANE_RUN_SPEC_SCHEMA_VERSION,
            "authority": (
                "pssolver.configuration.PlaneBerisEdwardsRunSpec"
            ),
            "runtime_path": run_spec.runtime_path.value,
            "canonical_sha256": run_spec.canonical_sha256(),
        }

    def runtime_selection_metadata(
        self,
        run_spec: PlaneBerisEdwardsRunSpec,
    ) -> dict[str, object]:
        """Return additive requested/effective runtime-path metadata."""

        return {
            "authority": (
                "pssolver.configuration.PlaneBerisEdwardsRunSpec.runtime_path"
            ),
            "requested": run_spec.runtime_path.value,
            "effective": run_spec.runtime_path.value,
            "default": DEFAULT_PLANE_RUNTIME_PATH,
            "fallback_allowed": False,
        }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER = (
    PlaneBerisEdwardsSchemaV1CompatibilitySerializer()
)


__all__ = [
    "PLANE_BERIS_EDWARDS_SCHEMA_V1_SERIALIZER",
    "PLANE_RUN_SPEC_SCHEMA_VERSION",
    "PlaneBerisEdwardsSchemaV1CompatibilitySerializer",
]
