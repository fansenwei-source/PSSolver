"""RC4.5.5 shared public validation and diagnostics-envelope contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pssolver.configuration.execution_device import validate_execution_device
from pssolver.configuration.public_option_validation import (
    validate_iterative_pressure_options,
    validate_public_execution_device,
)
from pssolver.workflows.diagnostic_io import diagnostics_metadata_envelope


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
RC457 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
)
RC457_RECOVERY = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
)
RC458 = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
RELEASE = ROOT / "notes/pssolver_v0_2_0_rc4.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"


def _latest_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        RC456,
        RC457,
        RC457_RECOVERY,
        RC458,
        RELEASE,
        POST_RELEASE,
        RC5_W1,
        RC5_R51,
        RC5_R52,
        RC5_R53,
        RC5_R55,
        RC5_R56,
        RC5_R57,
    ):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if (
            registration is not None
            and registration["historical_sha256"] == resolved
        ):
            resolved = registration["current_sha256"]
    return resolved


@pytest.mark.parametrize(
    "requested",
    ("auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2147483647"),
)
def test_public_device_tokens_accept_only_the_frozen_canonical_grammar(
    requested,
):
    assert validate_execution_device(requested) == requested


@pytest.mark.parametrize(
    "requested",
    (
        "",
        " cpu",
        "cpu ",
        "cpu:0",
        "cuda:-1",
        "cuda:+1",
        "cuda:01",
        "CUDA",
        "mps",
        0,
        None,
    ),
)
def test_public_device_tokens_reject_aliases_whitespace_and_non_strings(
    requested,
):
    with pytest.raises(ValueError, match="invalid execution device"):
        validate_execution_device(requested)


def test_public_device_normalization_records_requested_and_resolved(monkeypatch):
    monkeypatch.setattr(
        "pssolver.configuration.execution_device.torch.cuda.is_available",
        lambda: False,
    )
    assert validate_public_execution_device("auto") == {
        "requested_execution_device": "auto",
        "resolved_execution_device": "cpu",
    }


def _pressure(**overrides):
    value = {
        "algorithm": "preconditioned_conjugate_gradient",
        "relative_tolerance": 1.0e-10,
        "max_iterations": 40,
        "fixed_iterations": 12,
        "warm_start": True,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("fixed", (None, 1, 40))
def test_shared_pressure_validator_accepts_the_exact_contract(fixed):
    value = _pressure(fixed_iterations=fixed)
    assert validate_iterative_pressure_options(
        value,
        description="test pressure",
    ) == value


@pytest.mark.parametrize(
    "value",
    (
        _pressure(extra=True),
        {key: value for key, value in _pressure().items() if key != "warm_start"},
        _pressure(algorithm="direct"),
        _pressure(relative_tolerance=0.0),
        _pressure(relative_tolerance=float("nan")),
        _pressure(relative_tolerance=float("inf")),
        _pressure(relative_tolerance=True),
        _pressure(max_iterations=0),
        _pressure(max_iterations=True),
        _pressure(fixed_iterations=0),
        _pressure(fixed_iterations=41),
        _pressure(fixed_iterations=True),
        _pressure(warm_start="true"),
    ),
)
def test_shared_pressure_validator_rejects_every_noncanonical_form(value):
    with pytest.raises(ValueError):
        validate_iterative_pressure_options(
            value,
            description="test pressure",
        )


def test_diagnostics_envelope_is_self_describing_and_keeps_v02_aliases():
    dtype = np.dtype(
        [
            ("step", np.int64),
            ("pressure_iterations", np.int64),
            ("residual", np.float64),
        ]
    )
    metadata = diagnostics_metadata_envelope(
        application_family="example",
        runtime_path="example_runtime",
        enabled=True,
        steps=[0, 2],
        dtype=dtype,
    )
    assert metadata == {
        "schema_version": 1,
        "application_family": "example",
        "runtime_path": "example_runtime",
        "enabled": True,
        "count": 2,
        "steps": [0, 2],
        "artifact_paths": {
            "npy": "diagnostics.npy",
            "csv": "diagnostics.csv",
        },
        "payload_schema": [
            {"name": "step", "dtype": "int64"},
            {"name": "pressure_iterations", "dtype": "int64"},
            {"name": "residual", "dtype": "float64"},
        ],
        "payload_units": {
            "step": "step_index",
            "pressure_iterations": "count",
            "residual": "solver_native_dimensionless",
        },
        "npy": "diagnostics.npy",
        "csv": "diagnostics.csv",
    }


def test_disabled_diagnostics_still_declare_payload_schema_and_units():
    dtype = np.dtype([("step", np.int64), ("value", np.float64)])
    metadata = diagnostics_metadata_envelope(
        application_family="example",
        runtime_path="example_runtime",
        enabled=False,
        steps=[],
        dtype=dtype,
    )
    assert metadata["artifact_paths"] == {"npy": None, "csv": None}
    assert metadata["npy"] is None
    assert metadata["csv"] is None
    assert metadata["payload_schema"]
    assert set(metadata["payload_units"]) == set(dtype.names)


def test_disabled_diagnostics_reject_captured_steps():
    with pytest.raises(ValueError, match="cannot declare captured steps"):
        diagnostics_metadata_envelope(
            application_family="example",
            runtime_path="example_runtime",
            enabled=False,
            steps=[0],
            dtype=np.dtype([("step", np.int64)]),
        )


def test_rc455_record_closes_only_b13_b15_b17_and_binds_sources():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_RC4_5_5_PUBLIC_VALIDATION_OBSERVABILITY_CPU_CANDIDATE"
    )
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["issues"] == {
        "B13": "closed_verified",
        "B15": "closed_verified",
        "B17": "closed_verified",
    }
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == (
            _latest_successor(relative, expected)
        )
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == (
            _latest_successor(relative, registration["current_sha256"])
        )
    assert record["verification"]["focused_public_tests"] == {
        "passed": 200,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert record["verification"]["full_cpu_suite"] == {
        "passed": 3348,
        "failed": 0,
        "skipped": 0,
        "deselected": 1,
        "subtests_passed": 8,
        "deselected_node_id": (
            "tests/test_rc4_plane_nyquist_cuda.py::"
            "test_plane_nyquist_repair_binds_allocated_cuda_device_and_"
            "storage_equivalence"
        ),
        "deselected_reason": (
            "pre-existing CUDA-only test reserved for an explicitly "
            "authorized H100 qualification"
        ),
    }
    assert record["scope"]["B11_modified"] is False
    assert record["scope"]["scientific_model_equations_changed"] is False
    assert record["scope"]["diagnostic_payload_bytes_changed"] is False
    assert record["authorization"]["rc4_5_5_complete"] is True
    assert record["authorization"]["eligible_for_rc4_5_6_implementation"] is True
    assert record["authorization"]["rc4_5_6_implementation_authorized"] is False

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    issues = {entry["id"]: entry for entry in matrix["items"]}
    assert issues["B13"]["state"] == "partial_residual"
    assert issues["B17"]["state"] == "partial_residual"
    assert issues["B15"]["state"] == "closed_verified"
    progress = matrix["progress"]
    assert progress["rc4_5_5_complete"] is True
    assert progress["rc4_5_5_record_sha256"] == hashlib.sha256(
        RECORD.read_bytes()
    ).hexdigest()
    assert progress["eligible_for_rc4_5_6_implementation"] is True
