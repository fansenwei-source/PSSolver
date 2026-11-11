from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pssolver.configuration.package_construction import plan_package_runtime_construction
from pssolver.configuration.public_complete_stress_validation import (
    validate_complete_stress_runtime_parameters,
    validate_existing_q_snapshot_shape,
)
from pssolver.configuration.public_option_validation import (
    ITERATIVE_PRESSURE_OPTION_KEYS,
    validate_iterative_pressure_options,
    validate_public_execution_device,
)
from pssolver.configuration.simulation import InitialConditionSource, SimulationSpec
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core._canonical import canonical_json
from pssolver.core._validation import require_exact_keys
from pssolver.core.domain import GridPlacement
from pssolver.core.errors import (
    SimulationInputError,
    SimulationInputFileNotFoundError,
    SimulationInputIOError,
)
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import SpectralStorage
from pssolver.core.pressure import channel_pressure_minimum_relative_tolerance_for_precision
from pssolver.core.terms import Parameter
from pssolver.io.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    checkpoint_directory_transaction,
    checkpoint_file_sha256,
    load_checkpoint_numpy,
    read_checkpoint_json,
    require_exact_checkpoint_keys,
    resolve_checkpoint_payload,
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
)
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    CheckpointIdentityLayer,
    IdentityLayerKind,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    resolve_legacy_identity_schema,
)
from pssolver.linear_solvers.stokes._pressure_policy import channel_pressure_policy_metadata
from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.models.active_nematics.fields import CHANNEL_Q_BOUNDARY_CONDITIONS
from pssolver.planning.assemble import (
    AssemblyProblem,
    EulerInput,
    ExecutionInput,
    GeometryInput,
    NumericsInput,
    OutputInput,
    assemble,
)
from pssolver.presets.beris_edwards import CompleteStressBerisEdwards


CHECKPOINT_VERSION = 3


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsCheckpointHeader:
    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    compatibility_identity: CheckpointCompatibilityIdentity | None = None

    @property
    def compatibility_identity_sha256(self) -> str | None:
        identity = self.compatibility_identity
        return None if identity is None else identity.canonical_sha256()


LEGACY_CHECKPOINT_VERSION = 2

LEGACY_CHECKPOINT_VERSION_V1 = 1

_STATE_COMPONENTS = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")

_SUPPORTED_CHECKPOINT_VERSIONS = frozenset(
    {LEGACY_CHECKPOINT_VERSION_V1, LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION}
)


def _channel_backend_restart_identity(
    backend_restart: dict[str, object],
) -> CheckpointIdentityLayer:
    normalized = json.loads(
        canonical_json(dict(backend_restart))
    )
    normalized.pop("runtime_identity_sha256", None)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.BACKEND_RESTART,
        version="channel.production.backend.v1",
        payload=normalized,
    )


def _channel_run_spec_dynamics(
    run_spec: ChannelBerisEdwardsRunSpec,
) -> dict[str, object]:
    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a ChannelBerisEdwardsRunSpec")
    simulation = run_spec.simulation
    execution = simulation.execution.to_metadata()
    execution["options"].pop("device")
    return {
        "scientific": simulation.scientific_identity_metadata(),
        "discretization": simulation.discretization_identity_metadata(),
        "execution": execution,
    }


def _channel_state_layout_identity(
    fields: Mapping[str, torch.Tensor],
    pressure_guess: torch.Tensor,
) -> CheckpointIdentityLayer:
    def record(value: torch.Tensor) -> dict[str, object]:
        return {"shape": list(value.shape), "dtype": str(value.dtype)}

    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="channel.production.state.v3",
        payload={
            "component_order": list(_STATE_COMPONENTS),
            "spatial": {
                name: record(fields[name]) for name in _STATE_COMPONENTS
            },
            "spectral": {
                name: record(fields[f"{name}.hat"])
                for name in _STATE_COMPONENTS
            },
            "persistent": {"pressure_guess": record(pressure_guess)},
        },
    )


def _channel_state_layout_identity_from_metadata(
    metadata: Mapping[str, object],
) -> CheckpointIdentityLayer:
    manifest = metadata.get("tensor_files")
    backend_files = metadata.get("backend_files")
    if not isinstance(manifest, Mapping) or not isinstance(
        backend_files, Mapping
    ):
        raise CheckpointIntegrityError(
            "Channel checkpoint tensor manifest is missing",
            operation="complete_stress_channel_checkpoint_import",
        )

    def layout(record: object, description: str) -> dict[str, object]:
        if not isinstance(record, Mapping):
            raise CheckpointIntegrityError(
                f"{description} tensor record is missing",
                operation="complete_stress_channel_checkpoint_import",
            )
        shape = record.get("shape")
        dtype = record.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in shape
            )
            or not isinstance(dtype, str)
        ):
            raise CheckpointIntegrityError(
                f"{description} tensor layout is invalid",
                operation="complete_stress_channel_checkpoint_import",
            )
        return {"shape": shape, "dtype": f"torch.{dtype}"}

    result: dict[str, object] = {
        "component_order": list(_STATE_COMPONENTS),
        "spatial": {},
        "spectral": {},
        "persistent": {},
    }
    for kind in ("spatial", "spectral"):
        records = manifest.get(kind)
        if not isinstance(records, Mapping) or set(records) != set(
            _STATE_COMPONENTS
        ):
            raise CheckpointIntegrityError(
                "Channel checkpoint tensor manifest is incomplete",
                operation="complete_stress_channel_checkpoint_import",
            )
        result[kind] = {
            name: layout(records[name], f"{kind} {name}")
            for name in _STATE_COMPONENTS
        }
    if set(backend_files) != {"pressure_guess"}:
        raise CheckpointIntegrityError(
            "Channel checkpoint backend manifest is incomplete",
            operation="complete_stress_channel_checkpoint_import",
        )
    result["persistent"] = {
        "pressure_guess": layout(
            backend_files["pressure_guess"],
            "pressure guess",
        )
    }
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="channel.production.state.v3",
        payload=result,
    )


def _legacy_channel_format_for_generation(
    generation: SourceReleaseGeneration,
) -> int:
    if not isinstance(generation, SourceReleaseGeneration):
        raise TypeError("source_release_generation must be a SourceReleaseGeneration")
    return (
        LEGACY_CHECKPOINT_VERSION_V1
        if generation is SourceReleaseGeneration.RC1
        else LEGACY_CHECKPOINT_VERSION
    )


def _load_checkpoint(directory, adapter, *, run_spec):
    directory = Path(directory).expanduser().resolve()
    metadata = _read_channel_checkpoint_metadata(directory)
    header = read_channel_beris_edwards_checkpoint_header(directory)
    if header.compatibility_identity is None:
        raise CheckpointCompatibilityError(
            "legacy Channel checkpoint requires explicit identity upgrade",
            operation="complete_stress_channel_checkpoint_import",
        )
    completed_steps, integrator = _validated_progress(metadata)
    target_refresh_interval = (
        adapter.solver.integrator.spectral_refresh_interval
    )
    if integrator["spectral_refresh_interval"] != target_refresh_interval:
        raise CheckpointCompatibilityError(
            "Channel checkpoint spectral refresh policy does not match target",
            operation="complete_stress_channel_checkpoint_import",
            details={
                "checkpoint_interval": integrator["spectral_refresh_interval"],
                "target_interval": target_refresh_interval,
            },
        )
    if header.compatibility_identity.state_layout != (
        _channel_state_layout_identity_from_metadata(metadata)
    ):
        raise CheckpointIntegrityError(
            "Channel state layout identity is inconsistent",
            operation="complete_stress_channel_checkpoint_import",
        )
    backend_metadata = metadata.get("backend_restart")
    if not isinstance(backend_metadata, dict) or (
        header.compatibility_identity.backend_restart
        != _channel_backend_restart_identity(backend_metadata)
    ):
        raise CheckpointIntegrityError(
            "Channel backend restart identity is inconsistent",
            operation="complete_stress_channel_checkpoint_import",
        )

    fields = adapter.fields
    pressure_guess = adapter.capture_pressure_guess()
    expected_identity = build_channel_checkpoint_compatibility_identity(
        run_spec,
        fields=fields,
        pressure_guess=pressure_guess,
        backend_restart=adapter.backend_restart_metadata(),
    )
    if header.compatibility_identity != expected_identity:
        raise CheckpointCompatibilityError(
            "Channel compatibility identity does not match target",
            operation="complete_stress_channel_checkpoint_import",
        )
    pending = []
    manifest = metadata.get("tensor_files", {})
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        records = manifest.get(kind, {})
        if set(records) != set(_STATE_COMPONENTS):
            raise CheckpointIntegrityError(
                "Channel checkpoint tensor manifest is incomplete",
                operation="complete_stress_channel_checkpoint_import",
            )
        for name in _STATE_COMPONENTS:
            target = fields[f"{name}{suffix}"]
            values = _validated_array(
                directory,
                records[name],
                target,
                f"{kind} {name}",
                f"{kind}__{name}.npy",
            )
            pending.append((target, values))
    pressure_target = fields["p.hat"]
    pressure_record = metadata.get("backend_files", {}).get("pressure_guess")
    pressure_values = _validated_array(
        directory,
        pressure_record,
        pressure_target,
        "pressure guess",
        "backend__pressure_guess.npy",
    )

    materialized = tuple(
        (target, torch.from_numpy(values).to(device=target.device))
        for target, values in pending
    )
    materialized_pressure = torch.from_numpy(pressure_values).to(
        device=pressure_target.device
    )

    # Mutation starts only after every identity, file, shape, dtype, checksum,
    # finite-value, and device-materialization gate has passed.
    for target, values in materialized:
        target.copy_(values)
    adapter.restore_pressure_guess(materialized_pressure)
    adapter.restore_progress(
        completed_steps=completed_steps,
        spectral_refresh_interval=integrator["spectral_refresh_interval"],
        integrator_step_count=integrator["step_count"],
        integrator_refresh_count=integrator["refresh_count"],
    )
    adapter.restore_derived_state()
    return completed_steps


def _read_channel_checkpoint_metadata(directory: Path) -> dict[str, object]:
    metadata = read_checkpoint_json(
        directory / "checkpoint.json",
        operation="complete_stress_channel_checkpoint_import",
    )
    format_version = metadata.get("format_version")
    if (
        not isinstance(format_version, int)
        or isinstance(format_version, bool)
        or format_version not in _SUPPORTED_CHECKPOINT_VERSIONS
    ):
        raise CheckpointCompatibilityError(
            "unsupported complete-stress Channel checkpoint",
            operation="complete_stress_channel_checkpoint_import",
        )
    verify_checkpoint_metadata(
        metadata,
        required=format_version in {LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION},
        operation="complete_stress_channel_checkpoint_import",
    )
    if format_version == CHECKPOINT_VERSION:
        require_exact_checkpoint_keys(
            metadata,
            required={
                "format_version",
                "runtime_path",
                "run_provenance",
                "compatibility_identity",
                "compatibility_identity_sha256",
                "completed_steps",
                "integrator",
                "backend_restart",
                "tensor_files",
                "backend_files",
                "metadata_sha256",
            },
            optional={"migration_provenance"},
            description="complete-stress Channel checkpoint metadata",
            operation="complete_stress_channel_checkpoint_import",
            require_all=False,
        )
        require_exact_checkpoint_keys(
            metadata.get("run_provenance"),
            required={"legacy_runtime_identity_sha256"},
            description="complete-stress Channel run provenance",
            operation="complete_stress_channel_checkpoint_import",
        )
        tensor_files = require_exact_checkpoint_keys(
            metadata.get("tensor_files"),
            required={"spatial", "spectral"},
            description="complete-stress Channel tensor-file manifest",
            operation="complete_stress_channel_checkpoint_import",
        )
        for kind in ("spatial", "spectral"):
            require_exact_checkpoint_keys(
                tensor_files.get(kind),
                required=set(_STATE_COMPONENTS),
                description=f"complete-stress Channel {kind} manifest",
                operation="complete_stress_channel_checkpoint_import",
            )
        require_exact_checkpoint_keys(
            metadata.get("backend_files"),
            required={"pressure_guess"},
            description="complete-stress Channel backend manifest",
            operation="complete_stress_channel_checkpoint_import",
        )
    if metadata.get("runtime_path") != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        raise CheckpointCompatibilityError(
            "checkpoint runtime identity does not match target",
            operation="complete_stress_channel_checkpoint_import",
        )
    return metadata


def _sha256(path: Path) -> str:
    return checkpoint_file_sha256(
        path,
        operation="complete_stress_channel_checkpoint_io",
    )


def _validate_legacy_channel_upgrade_contract(
    metadata: Mapping[str, object],
    *,
    source_run_spec: ChannelBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    if not isinstance(source_run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("source_run_spec must be a ChannelBerisEdwardsRunSpec")
    expected_format = _legacy_channel_format_for_generation(
        source_release_generation
    )
    if metadata.get("format_version") != expected_format:
        raise ValueError("legacy Channel format does not match source release")
    if (
        metadata.get("runtime_identity_sha256")
        != source_run_spec.runtime_identity_sha256()
    ):
        raise ValueError(
            "legacy Channel checkpoint opaque identity is not authenticated"
        )
    # RC4.2.1 froze the Channel identity schema as v1 for every release;
    # the transport container independently became sealed format v2 in rc2.
    schema = resolve_legacy_identity_schema(
        LegacyIdentityKey(
            checkpoint_family=CheckpointFamily.CHANNEL_WORKFLOW,
            format_version=LEGACY_CHECKPOINT_VERSION_V1,
            source_release_generation=source_release_generation,
            runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
            applicability_class="configuration_dependent",
        )
    )
    periodic_length = source_run_spec.simulation.geometry.domain.shape[0]
    if source_release_generation is SourceReleaseGeneration.RC1:
        if periodic_length % 2 == 0:
            raise ValueError(
                "rc1 Channel checkpoint has pre-RC4.1 forward Nyquist dynamics"
            )
        raise ValueError(
            "rc1 Channel checkpoint predates the B2 incompressibility "
            "dynamics and cannot be upgraded for exact restart"
        )
    return schema, periodic_length


def _validated_array(
    directory,
    record,
    target,
    description,
    expected_filename,
):
    record = require_exact_checkpoint_keys(
        record,
        required={"file", "shape", "dtype", "sha256"},
        description=f"{description} record",
        operation="complete_stress_channel_checkpoint_import",
    )
    path = resolve_checkpoint_payload(
        directory,
        record["file"],
        operation="complete_stress_channel_checkpoint_import",
        description=description,
        expected_filename=expected_filename,
    )
    if _sha256(path) != record["sha256"]:
        raise CheckpointIntegrityError(
            f"{description} checksum mismatch",
            operation="complete_stress_channel_checkpoint_import",
        )
    try:
        values = load_checkpoint_numpy(
            path,
            operation="complete_stress_channel_checkpoint_import",
        )
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"{description} payload is invalid",
            operation="complete_stress_channel_checkpoint_import",
            legacy_exception_type=type(exc).__name__,
        ) from exc
    if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
        raise CheckpointIntegrityError(
            f"{description} metadata differs",
            operation="complete_stress_channel_checkpoint_import",
        )
    if tuple(values.shape) != tuple(target.shape):
        raise CheckpointIntegrityError(
            f"{description} shape differs",
            operation="complete_stress_channel_checkpoint_import",
        )
    if str(values.dtype) != str(target.detach().cpu().numpy().dtype):
        raise CheckpointIntegrityError(
            f"{description} dtype differs",
            operation="complete_stress_channel_checkpoint_import",
        )
    if not np.isfinite(values).all():
        raise CheckpointIntegrityError(
            f"{description} is not finite",
            operation="complete_stress_channel_checkpoint_import",
        )
    return values


def _validated_progress(metadata):
    completed_steps = metadata.get("completed_steps")
    if (
        not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or completed_steps < 0
    ):
        raise CheckpointIntegrityError(
            "checkpoint completed_steps is invalid",
            operation="complete_stress_channel_checkpoint_import",
        )
    integrator = metadata.get("integrator")
    if not isinstance(integrator, dict) or set(integrator) != {
        "spectral_refresh_interval",
        "step_count",
        "refresh_count",
    }:
        raise CheckpointIntegrityError(
            "checkpoint integrator metadata differs",
            operation="complete_stress_channel_checkpoint_import",
        )
    interval = integrator["spectral_refresh_interval"]
    if interval is not None and (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise CheckpointIntegrityError(
            "checkpoint spectral refresh interval is invalid",
            operation="complete_stress_channel_checkpoint_import",
        )
    for name in ("step_count", "refresh_count"):
        value = integrator[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise CheckpointIntegrityError(
                f"checkpoint {name} is invalid",
                operation="complete_stress_channel_checkpoint_import",
            )
    if interval is None:
        expected_refresh_count = 0
        expected_step_count = completed_steps
    else:
        expected_refresh_count, expected_step_count = divmod(
            completed_steps,
            interval,
        )
    if (
        integrator["step_count"] != expected_step_count
        or integrator["refresh_count"] != expected_refresh_count
    ):
        raise CheckpointIntegrityError(
            "checkpoint progress counters differ",
            operation="complete_stress_channel_checkpoint_import",
        )
    return completed_steps, integrator


def build_channel_checkpoint_compatibility_identity(
    run_spec: ChannelBerisEdwardsRunSpec,
    *,
    fields: Mapping[str, torch.Tensor],
    pressure_guess: torch.Tensor,
    backend_restart: dict[str, object],
) -> CheckpointCompatibilityIdentity:
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_WORKFLOW,
        runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        forward_dynamics=channel_forward_dynamics_identity(run_spec),
        state_layout=_channel_state_layout_identity(fields, pressure_guess),
        backend_restart=_channel_backend_restart_identity(backend_restart),
    )


def channel_forward_dynamics_identity(
    run_spec: ChannelBerisEdwardsRunSpec,
) -> CheckpointIdentityLayer:
    pressure = dict(
        run_spec.simulation.discretization_parameters["pressure_solver"]
    )
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="channel.production.forward.v4",
        payload={
            "run_spec_dynamics": _channel_run_spec_dynamics(run_spec),
            "pressure_solver_policy": channel_pressure_policy_metadata(
                relative_tolerance=pressure["relative_tolerance"],
                max_iterations=pressure["max_iterations"],
                fixed_iterations=pressure["fixed_iterations"],
                initial_guess_policy=(
                    "persistent_production_warm_start"
                ),
            ),
            "pressure_convergence_schema_version": 2,
            "dynamics_versions": {
                "q_evolution": "complete_stress_beris_edwards.v1",
                "forward_stokes_nyquist": (
                    "channel_no_slip.periodic_nyquist_zero.v2"
                ),
                "pressure_warm_start": "pcg_persistent_guess.v2",
                "pressure_policy": "channel_pressure_policy.v1",
                "spectral_refresh": "projected_euler_refresh.v1",
            },
        },
    )


def read_channel_beris_edwards_checkpoint_header(
    directory,
) -> ChannelBerisEdwardsCheckpointHeader:
    directory = Path(directory).expanduser().resolve()
    metadata = _read_channel_checkpoint_metadata(directory)
    format_version = metadata.get("format_version")
    completed_steps, _ = _validated_progress(metadata)
    compatibility_identity = None
    if format_version == CHECKPOINT_VERSION:
        run_provenance = metadata.get("run_provenance")
        if not isinstance(run_provenance, Mapping):
            raise CheckpointIntegrityError(
                "Channel checkpoint run provenance is missing",
                operation="complete_stress_channel_checkpoint_import",
            )
        runtime_identity = run_provenance.get("legacy_runtime_identity_sha256")
        try:
            compatibility_identity = (
                CheckpointCompatibilityIdentity.from_metadata(
                    metadata.get("compatibility_identity")
                )
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                "Channel compatibility identity is invalid",
                operation="complete_stress_channel_checkpoint_import",
                legacy_exception_type=type(exc).__name__,
            ) from exc
        expected = metadata.get("compatibility_identity_sha256")
        if compatibility_identity.canonical_sha256() != expected:
            raise CheckpointIntegrityError(
                "Channel compatibility identity checksum mismatch",
                operation="complete_stress_channel_checkpoint_import",
            )
        if compatibility_identity.family is not CheckpointFamily.CHANNEL_WORKFLOW:
            raise CheckpointIntegrityError(
                "Channel compatibility identity family differs",
                operation="complete_stress_channel_checkpoint_import",
            )
        if (
            compatibility_identity.runtime_path
            != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH
        ):
            raise CheckpointIntegrityError(
                "Channel compatibility runtime path differs",
                operation="complete_stress_channel_checkpoint_import",
            )
    else:
        runtime_identity = metadata.get("runtime_identity_sha256")
    if (
        not isinstance(runtime_identity, str)
        or len(runtime_identity) != 64
        or any(value not in "0123456789abcdef" for value in runtime_identity)
    ):
        raise CheckpointIntegrityError(
            "Channel runtime identity must be a lowercase SHA-256",
            operation="complete_stress_channel_checkpoint_import",
        )
    return ChannelBerisEdwardsCheckpointHeader(
        format_version=format_version,
        runtime_identity_sha256=runtime_identity,
        completed_steps=completed_steps,
        compatibility_identity=compatibility_identity,
    )


def read_channel_beris_edwards_checkpoint_progress(directory):
    """Read current authenticated progress without opening checkpoint tensors."""

    header = read_channel_beris_edwards_checkpoint_header(directory)
    if header.compatibility_identity is None:
        raise CheckpointCompatibilityError(
            "legacy Channel checkpoint requires explicit identity upgrade",
            operation="complete_stress_channel_checkpoint_import",
        )
    return header.completed_steps


def upgrade_channel_beris_edwards_checkpoint_directory(
    source_directory,
    target_directory,
    *,
    source_run_spec: ChannelBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    """Write a layered v3 copy after legacy forward-dynamics adjudication."""

    source = Path(source_directory).expanduser().resolve()
    target = Path(target_directory).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("legacy upgrade target must be outside the source checkpoint")
    metadata = _read_channel_checkpoint_metadata(source)
    if metadata.get("format_version") == CHECKPOINT_VERSION:
        raise ValueError("current Channel checkpoint does not require upgrade")
    schema, periodic_length = _validate_legacy_channel_upgrade_contract(
        metadata,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    completed_steps, _ = _validated_progress(metadata)
    state_layout = _channel_state_layout_identity_from_metadata(metadata)
    backend = metadata.get("backend_restart")
    if not isinstance(backend, dict):
        raise ValueError("Channel backend restart metadata is missing")
    compatibility_identity = CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_WORKFLOW,
        runtime_path=CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        forward_dynamics=channel_forward_dynamics_identity(source_run_spec),
        state_layout=state_layout,
        backend_restart=_channel_backend_restart_identity(backend),
    )
    with checkpoint_directory_transaction(
        target,
        operation="complete_stress_channel_checkpoint_migration",
    ) as (staging, target):
        shutil.copytree(source, staging, dirs_exist_ok=True)
        # A migration owns its private staging copy.  Make only that copy
        # writable so a read-only legacy source remains untouched while its
        # metadata can be upgraded and any failed transaction can be removed.
        for root, directories, files in os.walk(staging):
            root_path = Path(root)
            root_path.chmod(
                root_path.lstat().st_mode
                | stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IXUSR
            )
            for name in directories:
                path = root_path / name
                path.chmod(
                    path.lstat().st_mode
                    | stat.S_IRUSR
                    | stat.S_IWUSR
                    | stat.S_IXUSR
                )
            for name in files:
                path = root_path / name
                path.chmod(path.lstat().st_mode | stat.S_IRUSR | stat.S_IWUSR)
        upgraded = dict(metadata)
        upgraded.pop("metadata_sha256", None)
        upgraded.pop("runtime_identity_sha256", None)
        upgraded["format_version"] = CHECKPOINT_VERSION
        upgraded["run_provenance"] = {
            "legacy_runtime_identity_sha256": source_run_spec.runtime_identity_sha256()
        }
        upgraded["compatibility_identity"] = compatibility_identity.to_metadata()
        upgraded["compatibility_identity_sha256"] = (
            compatibility_identity.canonical_sha256()
        )
        upgraded["migration_provenance"] = {
            "kind": "channel_checkpoint_identity_upgrade_to_v3",
            "source_release_generation": source_release_generation.value,
            "source_commit": schema.source_commit,
            "source_format_version": metadata["format_version"],
            "legacy_schema_id": schema.schema_id,
            "legacy_canonicalizer_id": schema.canonicalizer_id,
            "periodic_axis_length": periodic_length,
            "forward_nyquist_applicability": (
                "inapplicable_odd_periodic_axis"
                if source_release_generation is SourceReleaseGeneration.RC1
                else "source_is_post_nyquist_repair"
            ),
            "source_completed_steps": completed_steps,
            "source_checkpoint_mutated": False,
        }
        upgraded = seal_checkpoint_metadata(upgraded)
        (staging / "checkpoint.json").write_text(
            json.dumps(upgraded, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


CHANNEL_COMPLETE_STRESS_RUNTIME_PATH = "channel_complete_stress"


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsRunSpec:
    """Exact public declaration consumed by the P8.3 application."""

    simulation: SimulationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        if self.simulation.equation_system.variant != (
            "complete_stress_beris_edwards"
        ):
            raise ValueError("Channel request requires complete-stress equations")
        if self.simulation.geometry.name != "rectangular_channel":
            raise ValueError("Channel request requires RectangularChannel")
        if self.simulation.execution.runtime_path != (
            CHANNEL_COMPLETE_STRESS_RUNTIME_PATH
        ):
            raise ValueError("Channel request has the wrong runtime path")

    @property
    def runtime_path(self) -> str:
        return CHANNEL_COMPLETE_STRESS_RUNTIME_PATH

    @property
    def output_dir(self) -> Path:
        return Path(self.simulation.workflow.options["output_dir"])

    @property
    def dry_run(self) -> bool:
        value = self.simulation.invocation.options.get("dry_run", False)
        if not isinstance(value, bool):
            raise TypeError("Channel dry_run must be a bool")
        return value

    def canonical_sha256(self) -> str:
        return self.simulation.canonical_sha256()

    def runtime_identity_sha256(self) -> str:
        identities = self.simulation.identity_metadata()
        payload = {
            name: identities[name]
            for name in ("scientific", "discretization", "execution")
        }
        encoded = canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


_EXECUTION_KEYS = frozenset(
    {
        "device",
        "disable_q_gradient_reuse",
        "fallback_allowed",
        "molecular_field_linear_space",
        "pointwise_execution",
        "stress_divergence_sum_space",
        "tf32",
    }
)

_WORKFLOW_KEYS = frozenset(
    {
        "checkpoint_interval",
        "diagnostic_interval",
        "diagnostics",
        "output_dir",
        "restart_from",
        "save_hydrodynamics",
        "save_interval",
        "save_start_step",
    }
)


def _exact_keys(value, expected, description, *, code):
    from .public_simulation_runner import _reject

    return require_exact_keys(
        value,
        expected,
        description,
        reject=lambda message: _reject(message, code=code),
    )


def _reject_input(
    message: str,
    *,
    path: Path,
    reason: str,
    exception: Exception | None = None,
) -> None:
    error_type = (
        SimulationInputFileNotFoundError
        if isinstance(exception, FileNotFoundError)
        else SimulationInputIOError
        if isinstance(exception, (OSError, ValueError))
        else SimulationInputError
    )
    error = error_type(
        message,
        application_family="channel_complete_stress_beris_edwards",
        stage="initial_condition",
        step=0,
        details={
            "path": str(path),
            "reason": reason,
            "legacy_exception_type": (
                None if exception is None else type(exception).__name__
            ),
        },
    )
    if exception is None:
        raise error
    raise error from exception


def _snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_channel_initial_q(run_spec: ChannelBerisEdwardsRunSpec):
    """Load the canonical Channel Q snapshot without constructing a runtime."""
    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a ChannelBerisEdwardsRunSpec")
    spec = run_spec.simulation
    snapshot = spec.initial_condition.parameters
    path = Path(snapshot["directory"]).expanduser().resolve() / f"Q_{snapshot['step']}.npy"
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        _reject_input(
            "Channel Q snapshot could not be read",
            path=path,
            reason="snapshot_unreadable",
            exception=exc,
        )
    shape = tuple(spec.geometry.domain.shape)
    if values.shape == (*shape, len(Q_COMPONENTS)):
        values = np.moveaxis(values, -1, 0)
    elif values.shape != (len(Q_COMPONENTS), *shape):
        _reject_input(
            "Channel Q snapshot must have shape (*grid, 5) or (5, *grid)",
            path=path,
            reason="snapshot_shape_mismatch",
        )
    expected_dtype = spec.numerics.precision.value
    if str(values.dtype) != expected_dtype:
        _reject_input(
            f"Channel Q snapshot dtype must be {expected_dtype}, got {values.dtype}",
            path=path,
            reason="snapshot_dtype_mismatch",
        )
    if not np.isfinite(values).all():
        _reject_input(
            "Channel Q snapshot contains NaN or Inf", path=path, reason="snapshot_nonfinite"
        )
    tensors = torch.from_numpy(np.array(values, copy=True))
    try:
        snapshot_sha256 = _snapshot_sha256(path)
    except OSError as exc:
        _reject_input(
            "Channel Q snapshot could not be hashed",
            path=path,
            reason="snapshot_unreadable",
            exception=exc,
        )
    return (
        {name: tensors[index] for index, name in enumerate(Q_COMPONENTS)},
        path,
        snapshot_sha256,
    )


def translate_channel(run_spec, *, flavour="production"):
    """Lower the validated v0.4 Channel declaration to a shared Engine plan."""
    if not isinstance(run_spec, ChannelBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a ChannelBerisEdwardsRunSpec")
    if flavour not in {"production", "functional_1_1"}:
        raise ValueError("unsupported Channel plan flavour")
    functional = flavour == "functional_1_1"
    spec = run_spec.simulation
    parameters = spec.equation_system.parameters
    execution = spec.execution.options
    numerics = spec.numerics
    domain = spec.geometry.domain
    pressure = spec.discretization_parameters["pressure_solver"]
    return assemble(
        AssemblyProblem(
            CompleteStressBerisEdwards(
                material=parameters["material"],
                ldg_l1=parameters["ldg_l1"],
                stokes=parameters["stokes"]["parameters"],
                alpha=Parameter(kind="field", controllable=True)
                if functional
                else Parameter(parameters["activity_amplitude"]),
            ),
            GeometryInput("rectangular_channel", tuple(domain.shape), tuple(domain.lengths)),
            {
                "Q": CHANNEL_Q_BOUNDARY_CONDITIONS,
                "velocity": dict.fromkeys(
                    VELOCITY_COMPONENTS, ("periodic", "dirichlet", "dirichlet")
                ),
                "pressure": ("periodic", "neumann", "neumann"),
            },
            EulerInput(spec.time_integration.integrator.dt),
            NumericsInput(
                numerics.spectral_storage.value,
                numerics.hermitian_axis,
                numerics.transform_execution_order.value,
                numerics.dealias_rule.value,
                numerics.projected_transform_execution.value,
            ),
            ExecutionInput(
                "eager" if functional else execution["pointwise_execution"],
                not functional and not execution["disable_q_gradient_reuse"],
                bool(spec.workflow.options["diagnostics"]),
                {
                    name: execution[name]
                    for name in ("molecular_field_linear_space", "stress_divergence_sum_space")
                },
                constraint_warm_start="zero" if functional else "previous",
                channel_pressure_options={
                    "pressure_relative_tolerance": pressure["relative_tolerance"],
                    "pressure_max_iterations": pressure["max_iterations"],
                    "pressure_fixed_iterations": pressure["fixed_iterations"],
                },
            ),
            OutputInput(
                ("Q", "velocity", "pressure"),
                "pssolver.runtime.channel_observations:capture_channel_observation",
            ),
        )
    ).plan


def compile_channel_beris_edwards_public_simulation(
    source: SimulationSpec,
    product_type,
):
    """Compile the single P8.3 Channel application without allocation."""

    from .public_simulation_runner import (
        PublicCompilationRejectionCode as Code,
    )
    from .public_simulation_runner import (
        _reject,
        _validate_invocation_options,
    )

    if (source.equation_system.variant, source.geometry.name) != (
        "complete_stress_beris_edwards",
        "rectangular_channel",
    ):
        raise AssertionError("complete-stress Channel compiler got wrong key")
    if source.geometry.domain.grid_placement is not GridPlacement.CELL_CENTERED:
        _reject(
            "complete-stress Channel currently requires a cell-centered grid",
            code=Code.GEOMETRY_CONTRACT,
        )
    if source.execution.backend != "torch_spectral":
        _reject(
            "complete-stress Channel requires torch_spectral",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.execution.runtime_path != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        _reject(
            "complete-stress Channel requires runtime_path='channel_complete_stress'",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject(
            "P8.3 currently requires projected Euler",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if source.initial_condition.source is not InitialConditionSource.SNAPSHOT:
        _reject(
            "P8.3 requires an external Q snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if source.initial_condition.family != "external_snapshot":
        _reject(
            "P8.3 snapshot family must be external_snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    snapshot = dict(source.initial_condition.parameters)
    if set(snapshot) != {"directory", "step", "mode"}:
        _reject(
            "P8.3 snapshot requires directory, step, and mode",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if snapshot["mode"] != "branch":
        _reject(
            "P8.3 accepts snapshot mode='branch'; restart uses checkpoints",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    try:
        validate_existing_q_snapshot_shape(
            snapshot,
            tuple(source.geometry.domain.shape),
            component_count=len(Q_COMPONENTS),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _reject(str(exc), code=Code.INITIAL_CONDITION_CONTRACT)

    execution = _exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "complete-stress Channel execution options",
        code=Code.EXECUTION_CONTRACT,
    )
    if execution["fallback_allowed"] is not False:
        _reject(
            "complete-stress Channel fallback must remain disabled",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["tf32"] != "off":
        _reject(
            "P8.3 qualifies strict TF32-off execution only",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["molecular_field_linear_space"] not in {
        "physical",
        "spectral",
    }:
        _reject(
            "unsupported Channel molecular-field linear space",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["stress_divergence_sum_space"] != "physical":
        _reject(
            "two-bounded-axis complete stress requires physical-space "
            "component-basis divergence summation",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["pointwise_execution"] not in {"eager", "compile"}:
        _reject(
            "unsupported Channel pointwise execution mode",
            code=Code.EXECUTION_CONTRACT,
        )
    try:
        device_normalization = validate_public_execution_device(execution["device"])
    except ValueError as exc:
        _reject(str(exc), code=Code.EXECUTION_CONTRACT)
    if not isinstance(execution["disable_q_gradient_reuse"], bool):
        _reject(
            "Channel Q-gradient reuse flag must be a bool",
            code=Code.EXECUTION_CONTRACT,
        )
    _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "complete-stress Channel workflow options",
        code=Code.OUTPUT_CONTRACT,
    )
    _validate_invocation_options(
        source.invocation.options,
        description="Channel invocation options",
    )
    discretization = _exact_keys(
        source.discretization_parameters,
        frozenset({"pressure_solver"}),
        "complete-stress Channel discretization",
        code=Code.DISCRETIZATION_CONTRACT,
    )
    pressure = _exact_keys(
        discretization["pressure_solver"],
        ITERATIVE_PRESSURE_OPTION_KEYS,
        "Channel pressure solver",
        code=Code.DISCRETIZATION_CONTRACT,
    )
    try:
        pressure = validate_iterative_pressure_options(
            pressure,
            description="Channel pressure solver",
            minimum_relative_tolerance=(
                channel_pressure_minimum_relative_tolerance_for_precision(
                    source.numerics.precision.value
                )
            ),
        )
    except ValueError as exc:
        _reject(str(exc), code=Code.DISCRETIZATION_CONTRACT)
    if pressure["warm_start"] is not True:
        _reject(
            "P8.3 requires pressure-PCG warm start",
            code=Code.DISCRETIZATION_CONTRACT,
        )
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject(
            "P8.3 requires disabled spectral refresh",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if source.numerics.spectral_storage is not SpectralStorage.FULL_COMPLEX:
        _reject(
            "P8.3 initially qualifies full_complex spectral storage",
            code=Code.NUMERICS_CONTRACT,
        )

    validate_complete_stress_runtime_parameters(source.equation_system.parameters)
    stokes = source.equation_system.parameters["stokes"]["parameters"]
    if stokes["pressure_gauge"] != "zero_mean":
        _reject(
            "Channel pressure requires the zero-mean gauge",
            code=Code.MODEL_CONTRACT,
        )

    run_spec = ChannelBerisEdwardsRunSpec(source)
    return product_type(
        application="channel_complete_stress_beris_edwards",
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=source,
        lowering_plan=lower_simulation_spec(source),
        construction_plan=plan_package_runtime_construction(source),
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "physical_values_changed": False,
            "pressure_gauge": "zero_mean",
            "declared_tangential_zero_mode_policy": stokes["tangential_zero_mode_policy"],
            "effective_tangential_zero_mode_policy": "not_applicable",
            "velocity_nullspace": ("absent_due_to_two_bounded_no_slip_axes"),
            "requested_molecular_field_linear_space": execution["molecular_field_linear_space"],
            "effective_molecular_field_linear_space": execution["molecular_field_linear_space"],
            "molecular_field_space_fallback_used": False,
            "runtime_fallback_allowed": False,
            **device_normalization,
            "pressure_solver_policy": channel_forward_dynamics_identity(run_spec).to_metadata()[
                "payload"
            ]["pressure_solver_policy"],
        },
    )
