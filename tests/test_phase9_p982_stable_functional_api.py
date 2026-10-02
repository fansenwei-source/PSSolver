"""Contracts for the P9.8.2 stable functional API and compatibility facade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver import __version__ as PSSOLVER_PACKAGE_VERSION
from pssolver import SpectralSolver
import pssolver.functional as compatibility
import pssolver.functional.api as stable
from pssolver.functional import (
    ChannelImplicitPressureAdjoint,
    ChannelPressureTransposeOperator,
    FunctionalCheckpointCompatibility,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)
from pssolver.functional.versioning import (
    runtime_identity_sha256_for_api_version,
)
from pssolver.functional.errors import translate_functional_exception
from pssolver.functional.pressure_metadata import (
    channel_pressure_solve_diagnostics,
)
from pssolver.io.checkpoint import (
    FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE,
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)
from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_9_p982_stable_functional_api.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _state_contract():
    physical = FunctionalTensorSpec(
        name="q_physical",
        shape=(5, 2, 2, 2),
        dtype="float64",
        device="cpu",
        batch_axis=1,
        layout="component_batch_xyz",
        meaning="test physical Q",
    )
    spectral = FunctionalTensorSpec(
        name="q_spectral",
        shape=(5, 2, 2, 2),
        dtype="complex128",
        device="cpu",
        batch_axis=1,
        layout="component_batch_kxyz",
        meaning="test spectral Q",
    )
    state_spec = FunctionalStateSpec(components=(physical, spectral))
    identity = FunctionalRuntimeIdentity(
        scientific={"model": "test"},
        discretization={"grid": [2, 2, 2]},
        execution={"runtime": "test"},
        state_layout=state_spec.to_metadata(),
    )
    state = (
        torch.arange(40, dtype=torch.float64).reshape(5, 2, 2, 2),
        torch.arange(40, dtype=torch.float64)
        .reshape(5, 2, 2, 2)
        .to(torch.complex128),
    )
    return state_spec, identity, state


def _periodic_identity(state_spec):
    return FunctionalRuntimeIdentity(
        scientific={"model": "test"},
        discretization={"grid": [2, 2, 2]},
        execution={
            "runtime": "test",
            "functional_runtime": {
                "kind": "periodic_activity_batch_one",
                "hermitian_state_projection": (
                    "self_conjugate_planes_each_step"
                ),
            },
        },
        state_layout=state_spec.to_metadata(),
    )


def _legacy_bridge_metadata(identity, bridge_metadata):
    result = dict(bridge_metadata)
    result["format_version"] = 1
    result["api_version"] = "0.1-provisional"
    result["functional_runtime_identity_sha256"] = (
        runtime_identity_sha256_for_api_version(
            identity.to_metadata(),
            "0.1-provisional",
        )
    )
    return result


def _write_legacy_checkpoint_metadata(directory, metadata):
    metadata["format_version"] = 1
    metadata.pop("metadata_sha256", None)
    (directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE).unlink()
    (directory / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_current_checkpoint_metadata(directory, metadata):
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    (directory / "checkpoint.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )


def test_stable_surface_is_exactly_the_machine_readable_contract():
    record = _record()
    assert stable.FUNCTIONAL_API_VERSION == "1.0"
    assert list(stable.__all__) == record["stable_module"]["exports"]
    assert "FunctionalCapabilities" not in stable.__all__
    assert compatibility.FunctionalCapabilities is stable.FunctionalCapabilitySet


def test_version_negotiation_is_exact_and_legacy_is_read_only():
    current = stable.negotiate_functional_api_version(
        "1.0", purpose="construction"
    )
    assert current.to_metadata() == {
        "requested": "1.0",
        "effective": "1.0",
        "purpose": "construction",
        "legacy": False,
        "compatibility_reader": None,
    }
    legacy = stable.negotiate_functional_api_version(
        "0.1-provisional", purpose="checkpoint_read"
    )
    assert legacy.legacy is True
    assert legacy.effective == "1.0"
    assert legacy.compatibility_reader
    with pytest.raises(stable.FunctionalVersionError) as error:
        stable.negotiate_functional_api_version(
            "0.1-provisional", purpose="construction"
        )
    assert error.value.code == "functional_version_error"
    assert error.value.to_metadata()["details"]["supported"] == ["1.0"]
    with pytest.raises(stable.FunctionalVersionError):
        stable.negotiate_functional_api_version(
            "2.0", purpose="checkpoint_read"
        )


def test_protocol_provenance_records_package_and_two_minor_alias_window():
    metadata = stable.functional_protocol_provenance()
    assert metadata["package"] == {
        "name": "pssolver",
        "version": PSSOLVER_PACKAGE_VERSION,
    }
    protocol = metadata["functional_protocol"]
    assert protocol["current"] == "1.0"
    assert protocol["construction_versions"] == ["1.0"]
    assert protocol["checkpoint_read_versions"] == [
        "1.0",
        "0.1-provisional",
    ]
    assert protocol["silent_fallback"] is False
    policy = metadata["compatibility_policy"]
    assert policy["version"] == 2
    assert policy["legacy_checkpoint_schema_repair"] is False
    assert policy["periodic_bridge_current_format_version"] == 2
    assert policy["legacy_identity_schemas"] == {
        "0.1-provisional": (
            "frozen_periodic_identity_before_hermitian_state_repair"
        )
    }
    alias = policy["FunctionalCapabilities_alias"]
    assert alias["deprecated_in_package_version"] == "0.2.0"
    assert alias["earliest_removal_package_version"] == "0.4.0"


def test_stable_errors_remain_compatible_with_builtin_exception_catches():
    with pytest.raises(stable.FunctionalTypeError) as error:
        stable.periodic_activity_functional_request(None)
    assert isinstance(error.value, TypeError)
    assert error.value.operation == "periodic_activity_functional_request"
    assert error.value.to_metadata()["code"] == "functional_type_error"

    with pytest.raises(stable.FunctionalValueError):
        FunctionalTensorSpec(
            name="q",
            shape=(),
            dtype="float64",
            device="cpu",
            batch_axis=0,
            layout="test",
            meaning="test",
        )

    checkpoint_error = translate_functional_exception(
        ValueError("malformed manifest"),
        operation="checkpoint_import",
    )
    assert isinstance(
        checkpoint_error,
        stable.FunctionalCheckpointIntegrityError,
    )
    assert isinstance(checkpoint_error, ValueError)

    checkpoint_io_error = translate_functional_exception(
        PermissionError("read-only checkpoint target"),
        operation="checkpoint_export",
    )
    assert type(checkpoint_io_error) is stable.FunctionalCheckpointError
    assert checkpoint_io_error.operation == "checkpoint_export"


def test_wrong_geometry_checkpoint_is_compatibility_not_integrity(tmp_path):
    from pssolver.functional import (
        ChannelActivityCheckpointBridge,
        PeriodicActivityCheckpointBridge,
    )

    state_spec, channel_identity, state = _state_contract()
    channel = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=channel_identity,
        production_runtime_identity_sha256="a" * 64,
        production_backend_restart={},
    )
    periodic = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=_periodic_identity(state_spec),
        production_runtime_identity_sha256="b" * 64,
        backend_restart={},
    )
    channel_directory = channel.export_checkpoint(
        tmp_path / "channel_for_periodic",
        state,
        completed_steps=0,
    )

    with pytest.raises(stable.FunctionalCheckpointCompatibilityError):
        periodic.import_checkpoint(channel_directory)


def test_channel_v1_legacy_checkpoint_reader_is_exact_and_machine_readable(
    tmp_path,
):
    from pssolver.functional import ChannelActivityCheckpointBridge

    state_spec, identity, state = _state_contract()
    bridge = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="a" * 64,
        production_backend_restart={},
    )
    directory = bridge.export_checkpoint(
        tmp_path / "channel", state, completed_steps=3
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["functional_bridge"] = _legacy_bridge_metadata(
        identity,
        metadata["functional_bridge"],
    )
    _write_legacy_checkpoint_metadata(directory, metadata)

    restored = bridge.import_checkpoint(directory)
    assert restored.source_format == "channel_functional_bridge_v1"
    assert isinstance(restored.compatibility, FunctionalCheckpointCompatibility)
    assert restored.compatibility.to_metadata() == {
        "source_api_version": "0.1-provisional",
        "target_api_version": "1.0",
        "reader": "qualified_v1_exact_schema_reader_for_0_1_provisional",
        "exact_current_protocol": False,
    }
    assert all(torch.equal(a, b) for a, b in zip(restored.state, state, strict=True))


def test_periodic_v1_legacy_checkpoint_reader_is_exact_and_machine_readable(
    tmp_path,
):
    from pssolver.functional import PeriodicActivityCheckpointBridge

    state_spec, _, state = _state_contract()
    identity = _periodic_identity(state_spec)
    bridge = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="b" * 64,
        backend_restart={},
    )
    directory = bridge.export_checkpoint(
        tmp_path / "periodic", state, completed_steps=4
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["functional_bridge"] = _legacy_bridge_metadata(
        identity,
        metadata["functional_bridge"],
    )
    historical_identity = identity.to_metadata()
    historical_identity["api_version"] = "0.1-provisional"
    historical_identity["execution"]["functional_runtime"].pop(
        "hermitian_state_projection"
    )
    expected_legacy_sha256 = hashlib.sha256(
        json.dumps(
            historical_identity,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert metadata["functional_bridge"][
        "functional_runtime_identity_sha256"
    ] == expected_legacy_sha256
    _write_legacy_checkpoint_metadata(directory, metadata)

    restored = bridge.import_checkpoint(directory)
    assert restored.source_format == "periodic_functional_bridge_v1"
    assert restored.compatibility.source_api_version == "0.1-provisional"
    assert restored.compatibility.target_api_version == "1.0"
    assert restored.compatibility.exact_current_protocol is False
    assert all(torch.equal(a, b) for a, b in zip(restored.state, state, strict=True))


def test_periodic_v1_current_checkpoint_reader_requires_current_identity(
    tmp_path,
):
    from pssolver.functional import PeriodicActivityCheckpointBridge

    state_spec, _, state = _state_contract()
    identity = _periodic_identity(state_spec)
    bridge = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="d" * 64,
        backend_restart={},
    )
    directory = bridge.export_checkpoint(
        tmp_path / "periodic-current-v1", state, completed_steps=4
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["functional_bridge"]["format_version"] = 1
    metadata["format_version"] = 1
    metadata.pop("metadata_sha256", None)
    (directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE).unlink()
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restored = bridge.import_checkpoint(directory)

    assert restored.source_format == "periodic_functional_bridge_v1"
    assert restored.compatibility.reader == (
        "current_periodic_functional_bridge_v1_reader"
    )
    assert restored.compatibility.exact_current_protocol is True

    pre_repair_identity = identity.to_metadata()
    pre_repair_identity["execution"]["functional_runtime"].pop(
        "hermitian_state_projection"
    )
    metadata["functional_bridge"][
        "functional_runtime_identity_sha256"
    ] = hashlib.sha256(
        json.dumps(
            pre_repair_identity,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(stable.FunctionalCheckpointCompatibilityError):
        bridge.import_checkpoint(directory)


def test_unknown_checkpoint_api_is_rejected_before_tensor_loading(tmp_path):
    from pssolver.functional import ChannelActivityCheckpointBridge

    state_spec, identity, state = _state_contract()
    bridge = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="c" * 64,
        production_backend_restart={},
    )
    directory = bridge.export_checkpoint(
        tmp_path / "unknown", state, completed_steps=0
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["functional_bridge"]["api_version"] = "9.9"
    _rewrite_current_checkpoint_metadata(directory, metadata)
    (directory / "state__q_physical.npy").unlink()
    with pytest.raises(stable.FunctionalVersionError):
        bridge.import_checkpoint(directory)


def test_pressure_metadata_distinguishes_cap_from_required_postcondition():
    spectral = SpectralSolver(
        (3, 2, 2),
        L=(3.0, 2.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-30,
        pressure_max_iterations=100,
        pressure_fixed_iterations=2,
    )
    operator = ChannelPressureTransposeOperator(solver)
    implicit = ChannelImplicitPressureAdjoint(solver)
    generator = torch.Generator().manual_seed(20260930)
    rhs = torch.complex(
        torch.randn(solver.pressure_null_mask.shape, generator=generator),
        torch.randn(solver.pressure_null_mask.shape, generator=generator),
    ).to(torch.complex128)
    rhs = operator._project_gauge(rhs)

    implicit.solve(rhs)
    metadata = implicit.pressure_solve_metadata()["primal"]
    required = {
        "solver_identity",
        "operator_identity",
        "requested_iteration_limit",
        "achieved_iteration_count",
        "achieved_absolute_residual",
        "achieved_relative_residual",
        "requested_relative_tolerance",
        "tolerance_semantics",
        "termination_reason",
        "acceptable",
    }
    assert required <= set(metadata)
    assert metadata["convergence_mode"] == (
        "deterministic_iteration_cap_with_early_convergence"
    )
    assert metadata["tolerance_semantics"] == (
        "early_convergence_criterion_not_required_at_cap"
    )
    assert metadata["requested_iteration_limit"] == 2
    assert metadata["achieved_iteration_count"] == 2
    assert metadata["termination_reason"] == "iteration_limit_reached"
    assert metadata["acceptable"] is True

    cap_edge = channel_pressure_solve_diagnostics(
        solver,
        operator_identity="channel_pressure_schur_primal",
        achieved_iteration_count=2,
        achieved_absolute_residual=1.0,
        achieved_relative_residual=1.0,
        termination_reason="preconditioned_residual_breakdown",
    )
    assert cap_edge.acceptable is True


def test_p982_record_does_not_authorize_consumer_migration_or_h100():
    authorization = _record()["authorization"]
    assert authorization["p9_8_2_complete"] is True
    assert authorization["p9_8_3_planning_eligible"] is True
    assert authorization["consumer_migration_complete"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["production_default_changed"] is False
    assert authorization["phase_9_complete"] is False
