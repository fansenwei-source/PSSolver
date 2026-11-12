"""Frozen v0.4 Periodic declarations and translation for Functional API 1.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from pssolver.core._canonical import canonical_json
from pssolver.core._validation import require_exact_keys
from pssolver.core.domain import GridPlacement
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import SpectralStorage
from pssolver.core.spectral_identity import periodic_hermitian_state_projection
from pssolver.errors import SimulationRestartError
from pssolver.functional.legacy_declarations.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.functional.legacy_declarations.public_complete_stress_validation import (
    validate_complete_stress_runtime_parameters,
    validate_existing_q_snapshot_shape,
)
from pssolver.functional.legacy_declarations.public_option_validation import (
    validate_public_execution_device,
)
from pssolver.functional.legacy_declarations.simulation import (
    InitialConditionSource,
    SimulationSpec,
)
from pssolver.functional.legacy_declarations.simulation_lowering import lower_simulation_spec
from pssolver.io.checkpoint import CheckpointCompatibilityError, CheckpointIntegrityError
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    CheckpointIdentityLayer,
    IdentityLayerKind,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    resolve_legacy_identity_schema,
)
from pssolver.io.initial_snapshot import load_q_snapshot
from pssolver.io.legacy.periodic_workflow import (
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PeriodicWorkflowCheckpoint,
    _legacy_periodic_format_for_generation,
    load_periodic_checkpoint,
    read_periodic_checkpoint_header,
    write_periodic_checkpoint,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.planning.engine_plan import Plan


PERIODIC_RUNTIME_PATH = "periodic_spectral"


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


@dataclass(frozen=True, slots=True)
class PeriodicBerisEdwardsRunSpec:
    """Exact public declaration consumed by the periodic application."""

    simulation: SimulationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        if self.simulation.equation_system.variant != (
            "complete_stress_beris_edwards"
        ):
            raise ValueError("periodic request requires complete-stress equations")
        if self.simulation.geometry.name != "periodic_box":
            raise ValueError("periodic request requires PeriodicBox geometry")
        if self.simulation.execution.runtime_path != PERIODIC_RUNTIME_PATH:
            raise ValueError("periodic request has the wrong runtime path")

    @property
    def runtime_path(self) -> str:
        return PERIODIC_RUNTIME_PATH

    @property
    def output_dir(self) -> Path:
        return Path(self.simulation.workflow.options["output_dir"])

    @property
    def dry_run(self) -> bool:
        value = self.simulation.invocation.options.get("dry_run", False)
        if not isinstance(value, bool):
            raise TypeError("periodic dry_run must be a bool")
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


def _exact_keys(value, expected, description, *, code):
    from .public_simulation_runner import _reject

    return require_exact_keys(
        value,
        expected,
        description,
        reject=lambda message: _reject(message, code=code),
    )


def compile_periodic_public_simulation(source: SimulationSpec, product_type):
    """Compile the single P8.2 periodic application without allocation."""

    from .public_simulation_runner import (
        PublicCompilationRejectionCode as Code,
        _reject,
        _validate_invocation_options,
    )

    if (source.equation_system.variant, source.geometry.name) != (
        "complete_stress_beris_edwards",
        "periodic_box",
    ):
        raise AssertionError("periodic compiler received the wrong registry key")
    if source.geometry.domain.grid_placement is not GridPlacement.CELL_CENTERED:
        _reject(
            "periodic execution currently requires a cell-centered grid",
            code=Code.GEOMETRY_CONTRACT,
        )
    if source.execution.backend != "torch_spectral":
        _reject(
            "periodic execution requires the torch_spectral backend",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.execution.runtime_path != PERIODIC_RUNTIME_PATH:
        _reject(
            "periodic execution requires runtime_path='periodic_spectral'",
            code=Code.EXECUTION_CONTRACT,
        )
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject(
            "periodic execution currently requires projected Euler",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if source.initial_condition.source is not InitialConditionSource.SNAPSHOT:
        _reject(
            "P8.2 periodic execution requires an external Q snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if source.initial_condition.family != "external_snapshot":
        _reject(
            "periodic snapshot family must be external_snapshot",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    snapshot = dict(source.initial_condition.parameters)
    if set(snapshot) != {"directory", "step", "mode"}:
        _reject(
            "periodic snapshot requires directory, step, and mode",
            code=Code.INITIAL_CONDITION_CONTRACT,
        )
    if snapshot["mode"] != "branch":
        _reject(
            "P8.2 accepts snapshot mode='branch'; restart uses checkpoints",
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
        "periodic execution options",
        code=Code.EXECUTION_CONTRACT,
    )
    if execution["fallback_allowed"] is not False:
        _reject(
            "periodic runtime fallback must remain disabled",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["tf32"] != "off":
        _reject(
            "P8.2 qualifies strict TF32-off execution only",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["molecular_field_linear_space"] not in {
        "physical",
        "spectral",
    }:
        _reject(
            "unsupported periodic molecular-field linear space",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["stress_divergence_sum_space"] not in {
        "physical",
        "spectral",
    }:
        _reject(
            "unsupported periodic stress-divergence sum space",
            code=Code.EXECUTION_CONTRACT,
        )
    if execution["pointwise_execution"] not in {"eager", "compile"}:
        _reject(
            "unsupported periodic pointwise execution mode",
            code=Code.EXECUTION_CONTRACT,
        )
    try:
        device_normalization = validate_public_execution_device(
            execution["device"]
        )
    except ValueError as exc:
        _reject(str(exc), code=Code.EXECUTION_CONTRACT)
    if not isinstance(execution["disable_q_gradient_reuse"], bool):
        _reject(
            "periodic Q-gradient reuse flag must be a bool",
            code=Code.EXECUTION_CONTRACT,
        )
    _exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "periodic workflow options",
        code=Code.OUTPUT_CONTRACT,
    )
    _validate_invocation_options(
        source.invocation.options,
        description="periodic invocation options",
    )
    if source.discretization_parameters:
        _reject(
            "periodic execution accepts no discretization overrides",
            code=Code.DISCRETIZATION_CONTRACT,
        )
    requested_velocity_policy = source.boundaries.declared_policy("velocity")
    if requested_velocity_policy is None:
        requested_velocity_policy = "qualified_periodic_wall_policy_unspecified"
    if requested_velocity_policy not in {
        "free_slip_velocity",
        "no_slip_velocity",
        "qualified_periodic_wall_policy_unspecified",
    }:
        _reject(
            "periodic velocity boundary declaration is not qualified",
            code=Code.BOUNDARY_CONTRACT,
        )
    if source.time_integration.refresh != {"mode": "disabled"}:
        _reject(
            "P8.2 periodic execution requires disabled spectral refresh",
            code=Code.TIME_INTEGRATION_CONTRACT,
        )
    if (
        source.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
        and source.numerics.hermitian_axis not in source.geometry.periodic_axes
    ):
        _reject(
            "periodic Hermitian packing must use a periodic axis",
            code=Code.NUMERICS_CONTRACT,
        )
    validate_complete_stress_runtime_parameters(
        source.equation_system.parameters
    )
    stokes = source.equation_system.parameters["stokes"]["parameters"]
    if stokes["pressure_gauge"] != "zero_mean":
        _reject(
            "periodic pressure requires the zero-mean gauge",
            code=Code.MODEL_CONTRACT,
        )
    zero_mode_policy = stokes["tangential_zero_mode_policy"]
    friction = stokes["friction"]
    if zero_mode_policy == "zero_mean":
        if friction != 0.0:
            _reject(
                "zero-mean periodic velocity requires zero friction",
                code=Code.MODEL_CONTRACT,
            )
        uniform_mode_action = "remove_all_uniform_velocity_and_force"
    elif zero_mode_policy == "friction":
        if friction <= 0.0:
            _reject(
                "periodic friction mode requires positive friction",
                code=Code.MODEL_CONTRACT,
            )
        uniform_mode_action = (
            "retain_all_uniform_velocity_resolved_by_friction"
        )
    else:
        _reject(
            "periodic velocity requires zero_mean or friction zero-mode policy",
            code=Code.MODEL_CONTRACT,
        )

    run_spec = PeriodicBerisEdwardsRunSpec(source)
    return product_type(
        application="periodic_complete_stress_beris_edwards",
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=source,
        lowering_plan=lower_simulation_spec(source),
        construction_plan=plan_package_runtime_construction(source),
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "physical_values_changed": False,
            "pressure_gauge": "zero_mean",
            "uniform_velocity_mode_action": uniform_mode_action,
            "requested_molecular_field_linear_space": execution[
                "molecular_field_linear_space"
            ],
            "effective_molecular_field_linear_space": execution[
                "molecular_field_linear_space"
            ],
            "molecular_field_space_fallback_used": False,
            "runtime_fallback_allowed": False,
            **device_normalization,
            "requested_velocity_boundary_policy": requested_velocity_policy,
            "effective_velocity_boundary_policy": "periodic_continuation",
            "velocity_boundary_policy_equivalence": (
                "qualified_periodic_no_face_equivalence"
            ),
        },
    )


def load_periodic_initial_q(run_spec: PeriodicBerisEdwardsRunSpec):
    """Load the validated compact-Q snapshot used by periodic runtimes.

    The helper is shared by the production application and the provisional
    functional runtime so both construction paths start from the same bytes,
    layout checks, and dtype contract.
    """

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    snapshot = spec.initial_condition.parameters
    return load_q_snapshot(snapshot['directory'], snapshot['step'], shape=tuple(spec.geometry.domain.shape), dtype=spec.numerics.precision.value, components=Q_COMPONENTS, label='periodic', application_family='periodic_complete_stress_beris_edwards')


@dataclass(frozen=True, slots=True)
class PeriodicTranslation:
    run_spec: PeriodicBerisEdwardsRunSpec
    plan: Plan
    initial_values: Mapping[str, torch.Tensor] | None
    snapshot_path: Path | None
    snapshot_sha256: str | None


def translate_periodic(spec: SimulationSpec, *, flavour, load_initial):
    """Apply the frozen validation before constructing a tensor-free plan."""
    from pssolver.core.terms import Parameter
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

    from .public_simulation_runner import PublicSimulationCompilation

    compiled = compile_periodic_public_simulation(spec, PublicSimulationCompilation)
    numerics = spec.numerics
    execution = spec.execution.options
    parameters = spec.equation_system.parameters
    if flavour not in {"production", "functional_1_1"}:
        raise ValueError("unsupported periodic plan flavour")
    functional = flavour == "functional_1_1"
    problem = AssemblyProblem(
        CompleteStressBerisEdwards(
            material=parameters["material"], ldg_l1=parameters["ldg_l1"],
            stokes=parameters["stokes"]["parameters"],
            alpha=Parameter(kind="field", controllable=True) if functional
            else Parameter(parameters["activity_amplitude"]),
        ),
        GeometryInput("periodic_box", tuple(spec.geometry.domain.shape), tuple(spec.geometry.domain.lengths)),
        {name: ("periodic",) * 3 for name in ("Q", "velocity", "pressure")},
        EulerInput(spec.time_integration.integrator.dt),
        NumericsInput(numerics.spectral_storage.value, numerics.hermitian_axis,
            numerics.transform_execution_order.value, numerics.dealias_rule.value,
            numerics.projected_transform_execution.value),
        ExecutionInput("eager" if functional else execution["pointwise_execution"],
            False if functional else not execution["disable_q_gradient_reuse"], not functional,
            {name: execution[name] for name in ("molecular_field_linear_space", "stress_divergence_sum_space")}),
        OutputInput(("Q", "velocity", "pressure"), "pssolver.runtime.observations:capture_periodic_observation"),
    )
    values, path, digest = (
        load_periodic_initial_q(compiled.run_spec) if load_initial else (None, None, None)
    )
    return PeriodicTranslation(compiled.run_spec, assemble(problem).plan, values, path, digest)


def _periodic_run_spec_dynamics(
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> dict[str, object]:
    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    simulation = run_spec.simulation
    execution = simulation.execution.to_metadata()
    execution["options"].pop("device")
    return {
        "scientific": simulation.scientific_identity_metadata(),
        "discretization": simulation.discretization_identity_metadata(),
        "execution": execution,
    }


def periodic_forward_dynamics_identity(
    run_spec: PeriodicBerisEdwardsRunSpec,
) -> CheckpointIdentityLayer:
    """Return periodic next-step semantics without allocation provenance."""

    storage = run_spec.simulation.numerics.spectral_storage.value
    hermitian_projection = periodic_hermitian_state_projection(storage)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.FORWARD_DYNAMICS,
        version="periodic.forward.v3",
        payload={
            "run_spec_dynamics": _periodic_run_spec_dynamics(run_spec),
            "dynamics_versions": {
                "q_evolution": "complete_stress_beris_edwards.v1",
                "hermitian_projection": hermitian_projection,
                "stokes_nyquist": "periodic_modal_stokes.nyquist_zero.v1",
                "spectral_refresh": "projected_euler_refresh.v1",
            },
        },
    )


def _periodic_state_layout_identity(
    evolved_spatial: Mapping[str, torch.Tensor],
    evolved_spectral: Mapping[str, torch.Tensor],
) -> CheckpointIdentityLayer:
    def layout(values: Mapping[str, torch.Tensor]) -> dict[str, object]:
        return {
            name: {
                "shape": list(values[name].shape),
                "dtype": str(values[name].dtype),
            }
            for name in Q_COMPONENTS
        }

    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.STATE_LAYOUT,
        version="periodic.production.state.v3",
        payload={
            "component_order": list(Q_COMPONENTS),
            "spatial": layout(evolved_spatial),
            "spectral": layout(evolved_spectral),
        },
    )


def _periodic_backend_restart_identity(
    backend_restart: Mapping[str, object],
) -> CheckpointIdentityLayer:
    normalized = json.loads(
        canonical_json(dict(backend_restart))
    )
    normalized.pop("runtime_identity_sha256", None)
    return CheckpointIdentityLayer(
        kind=IdentityLayerKind.BACKEND_RESTART,
        version="periodic.production.backend.v1",
        payload=normalized,
    )


def build_periodic_checkpoint_compatibility_identity(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    evolved_spatial: Mapping[str, torch.Tensor],
    evolved_spectral: Mapping[str, torch.Tensor],
    backend_restart: Mapping[str, object],
) -> CheckpointCompatibilityIdentity:
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PERIODIC_WORKFLOW,
        runtime_path=PERIODIC_RUNTIME_PATH,
        forward_dynamics=periodic_forward_dynamics_identity(run_spec),
        state_layout=_periodic_state_layout_identity(
            evolved_spatial,
            evolved_spectral,
        ),
        backend_restart=_periodic_backend_restart_identity(backend_restart),
    )


def _validate_periodic_embedded_identity(
    checkpoint: PeriodicWorkflowCheckpoint,
) -> None:
    identity = checkpoint.compatibility_identity
    if identity is None:
        raise CheckpointIntegrityError(
            "periodic compatibility identity is missing",
            operation="periodic_checkpoint_import",
        )
    if identity.state_layout != _periodic_state_layout_identity(
        checkpoint.evolved_spatial,
        checkpoint.evolved_spectral,
    ):
        raise CheckpointIntegrityError(
            "periodic state layout identity is inconsistent",
            operation="periodic_checkpoint_import",
        )
    if identity.backend_restart != _periodic_backend_restart_identity(
        checkpoint.backend_restart
    ):
        raise CheckpointIntegrityError(
            "periodic backend restart identity is inconsistent",
            operation="periodic_checkpoint_import",
        )


def _validate_legacy_periodic_upgrade_contract(
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
):
    if not isinstance(source_run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("source_run_spec must be a PeriodicBerisEdwardsRunSpec")
    expected_format = _legacy_periodic_format_for_generation(
        source_release_generation
    )
    if checkpoint.format_version != expected_format:
        raise ValueError("legacy periodic format does not match source release")
    if checkpoint.functional_bridge is not None:
        raise ValueError(
            "functional periodic checkpoint migration is deferred to RC4.2.4"
        )
    if (
        checkpoint.runtime_identity_sha256
        != source_run_spec.runtime_identity_sha256()
    ):
        raise ValueError(
            "legacy periodic checkpoint opaque identity is not authenticated"
        )
    schema = resolve_legacy_identity_schema(
        LegacyIdentityKey(
            checkpoint_family=CheckpointFamily.PERIODIC_WORKFLOW,
            format_version=expected_format,
            source_release_generation=source_release_generation,
            runtime_path=PERIODIC_RUNTIME_PATH,
            applicability_class="configuration_dependent",
        )
    )
    storage = source_run_spec.simulation.numerics.spectral_storage.value
    if (
        source_release_generation is SourceReleaseGeneration.RC1
        and storage == "full_complex"
    ):
        raise ValueError(
            "rc1 full-complex periodic checkpoint has pre-repair Hermitian dynamics"
        )
    return schema, storage


def upgrade_periodic_checkpoint(
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> PeriodicWorkflowCheckpoint:
    """Create a production v3 copy after explicit legacy adjudication."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be a PeriodicWorkflowCheckpoint")
    if checkpoint.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("current periodic checkpoint does not require upgrade")
    schema, storage = _validate_legacy_periodic_upgrade_contract(
        checkpoint,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    identity = build_periodic_checkpoint_compatibility_identity(
        source_run_spec,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
    )
    return PeriodicWorkflowCheckpoint(
        format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=checkpoint.runtime_identity_sha256,
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
        evolved_spatial=checkpoint.evolved_spatial,
        evolved_spectral=checkpoint.evolved_spectral,
        backend_restart=checkpoint.backend_restart,
        compatibility_identity=identity,
        migration_provenance={
            "kind": "periodic_checkpoint_identity_upgrade_to_v3",
            "source_release_generation": source_release_generation.value,
            "source_commit": schema.source_commit,
            "source_format_version": checkpoint.format_version,
            "legacy_schema_id": schema.schema_id,
            "legacy_canonicalizer_id": schema.canonicalizer_id,
            "spectral_storage": storage,
            "hermitian_repair_applicability": (
                "inapplicable_to_hermitian_half_storage"
                if source_release_generation is SourceReleaseGeneration.RC1
                else "source_is_post_repair"
            ),
            "source_checkpoint_mutated": False,
        },
    )


def upgrade_periodic_checkpoint_directory(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    source_run_spec: PeriodicBerisEdwardsRunSpec,
    source_release_generation: SourceReleaseGeneration,
) -> Path:
    """Write an authenticated v3 copy without changing the legacy source."""

    source = Path(source_directory).expanduser().resolve()
    target = Path(target_directory).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("legacy upgrade target must be outside the source checkpoint")
    header = read_periodic_checkpoint_header(source)
    if header.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("current periodic checkpoint does not require upgrade")
    # Authenticate the opaque header and adjudicate dynamics before payload load.
    shell = PeriodicWorkflowCheckpoint(
        format_version=header.format_version,
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=header.spectral_refresh_interval,
        integrator_step_count=header.integrator_step_count,
        integrator_refresh_count=header.integrator_refresh_count,
        evolved_spatial={name: torch.zeros(()) for name in Q_COMPONENTS},
        evolved_spectral={name: torch.zeros(()) for name in Q_COMPONENTS},
        backend_restart=header.backend_restart,
        functional_bridge=header.functional_bridge,
    )
    _validate_legacy_periodic_upgrade_contract(
        shell,
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    upgraded = upgrade_periodic_checkpoint(
        load_periodic_checkpoint(source),
        source_run_spec=source_run_spec,
        source_release_generation=source_release_generation,
    )
    return write_periodic_checkpoint(target, upgraded)


def validate_periodic_restore(
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    run_spec: PeriodicBerisEdwardsRunSpec,
    target_spatial: Mapping[str, torch.Tensor],
    target_spectral: Mapping[str, torch.Tensor],
    target_refresh_interval: int | None,
    backend_restart: Mapping[str, object],
) -> None:
    """Restore only after every identity, shape, and dtype gate has passed."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    if checkpoint.spectral_refresh_interval != target_refresh_interval:
        raise CheckpointCompatibilityError(
            "periodic checkpoint spectral refresh policy does not match target",
            operation="periodic_checkpoint_import",
            details={
                "checkpoint_interval": checkpoint.spectral_refresh_interval,
                "target_interval": target_refresh_interval,
            },
        )
    if checkpoint.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        _validate_periodic_embedded_identity(checkpoint)
        target_identity = build_periodic_checkpoint_compatibility_identity(
            run_spec,
            evolved_spatial=target_spatial,
            evolved_spectral=target_spectral,
            backend_restart=backend_restart,
        )
        if checkpoint.compatibility_identity != target_identity:
            raise CheckpointCompatibilityError(
                "periodic checkpoint compatibility identity does not match target",
                operation="periodic_checkpoint_import",
            )
    elif (
        checkpoint.format_version == PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        and checkpoint.functional_bridge is not None
        and checkpoint.functional_bridge.get("format_version") == 2
    ):
        # Preserve the already-qualified API-1.0 functional bridge until its
        # derivative-aware RC4.2.4 carrier is introduced.
        if checkpoint.runtime_identity_sha256 != run_spec.runtime_identity_sha256():
            raise CheckpointCompatibilityError(
                "checkpoint scientific/runtime identity does not match",
                operation="periodic_checkpoint_import",
            )
        if dict(checkpoint.backend_restart) != backend_restart:
            raise CheckpointCompatibilityError(
                "checkpoint backend contract does not match target",
                operation="periodic_checkpoint_import",
            )
    else:
        raise CheckpointCompatibilityError(
            "legacy periodic checkpoint requires explicit identity upgrade",
            operation="periodic_checkpoint_import",
        )
    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = target_spatial[name] if suffix == "" else target_spectral[name]
            if target.shape != source.shape:
                raise CheckpointCompatibilityError(
                    f"checkpoint tensor shape differs for {name}{suffix}",
                    operation="periodic_checkpoint_import",
                )
            if target.dtype != source.dtype:
                raise CheckpointCompatibilityError(
                    f"checkpoint tensor dtype differs for {name}{suffix}",
                    operation="periodic_checkpoint_import",
                )
            targets.append((target, source, f"{name}{suffix}"))


def _reject_restart(
    message: str,
    *,
    reason: str,
    step: int = 0,
    details: dict[str, object] | None = None,
    exception: Exception | None = None,
) -> None:
    error = SimulationRestartError(
        message,
        application_family="periodic_complete_stress_beris_edwards",
        stage="restart_validation",
        step=step,
        details={"reason": reason, **(details or {})},
    )
    if exception is None:
        raise error
    raise error from exception


def preflight_periodic_restart(directory, *, run_spec, plan, dtype):
    """Reject incompatible headers before payload reads or Engine construction."""
    from pssolver.io import engine_state
    from pssolver.io.checkpoint import read_checkpoint_json

    metadata = read_checkpoint_json(
        Path(directory) / "checkpoint.json", operation="periodic_checkpoint_import"
    )
    if metadata.get("format_kind") == "engine_state":
        header = engine_state.read_header(
            directory, plan_identity_sha256=plan.identity_sha256, dtype=dtype, batch=1
        )
        return "engine_state", header
    header = read_periodic_checkpoint_header(directory)
    if header.spectral_refresh_interval != plan.integrator.refresh_interval:
        raise CheckpointCompatibilityError(
            "periodic checkpoint spectral refresh policy does not match target",
            operation="periodic_checkpoint_import",
            details={
                "checkpoint_interval": header.spectral_refresh_interval,
                "target_interval": plan.integrator.refresh_interval,
            },
        )
    backend = {"kind": "periodic_stokes_stateless", "state_keys": []}
    if header.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        shape = plan.geometry.shape
        spectral_shape = list(shape)
        if plan.geometry.spectral_storage == "hermitian_half":
            axis = plan.geometry.hermitian_axis
            spectral_shape[axis] = spectral_shape[axis] // 2 + 1
        spatial = {name: SimpleNamespace(shape=(1, *shape), dtype=dtype) for name in Q_COMPONENTS}
        spectral = {
            name: SimpleNamespace(
                shape=(1, *spectral_shape),
                dtype=torch.complex64 if dtype == torch.float32 else torch.complex128,
            )
            for name in Q_COMPONENTS
        }
        target = build_periodic_checkpoint_compatibility_identity(
            run_spec,
            evolved_spatial=spatial,
            evolved_spectral=spectral,
            backend_restart=backend,
        )
        if header.compatibility_identity != target:
            raise CheckpointCompatibilityError(
                "periodic checkpoint compatibility identity does not match target",
                operation="periodic_checkpoint_import",
            )
    elif (
        header.format_version == PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        and header.functional_bridge is not None
        and header.functional_bridge.get("format_version") == 2
    ):
        if header.runtime_identity_sha256 != run_spec.runtime_identity_sha256():
            raise CheckpointCompatibilityError(
                "checkpoint scientific/runtime identity does not match",
                operation="periodic_checkpoint_import",
            )
        if dict(header.backend_restart) != backend:
            raise CheckpointCompatibilityError(
                "checkpoint backend contract does not match target",
                operation="periodic_checkpoint_import",
            )
    else:
        raise CheckpointCompatibilityError(
            "legacy periodic checkpoint requires explicit identity upgrade",
            operation="periodic_checkpoint_import",
        )
    return "periodic_workflow", header
