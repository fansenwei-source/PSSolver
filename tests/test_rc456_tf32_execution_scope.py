"""RC4.5.6 dynamic TF32 ownership and restoration contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration.execution_device import (
    configure_tf32_execution,
    scoped_tf32_execution,
    snapshot_tf32_execution_state,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
RC457 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
)
RC457_RECOVERY = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
)
RC458 = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
RELEASE = ROOT / "notes/pssolver_v0_2_0_rc4.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"


def _set_state(precision: str, matmul: bool, cudnn: bool) -> None:
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def _resolve_rc457_successor(relative: str, historical: str) -> str:
    resolved = historical
    for record_path in (RC457, RC457_RECOVERY, RC458, RELEASE, POST_RELEASE):
        successor = json.loads(record_path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
    return resolved


@pytest.fixture
def restore_tf32_state():
    before = snapshot_tf32_execution_state()
    yield before
    _set_state(
        before.torch_float32_matmul_precision,
        before.cuda_matmul_allow_tf32,
        before.cudnn_allow_tf32,
    )


def test_policy_description_and_invalid_policy_never_mutate_caller_state(
    restore_tf32_state,
):
    _set_state("high", True, False)
    caller = snapshot_tf32_execution_state()

    metadata = configure_tf32_execution(
        "off",
        real_dtype=torch.float64,
        device="cpu",
    )

    assert metadata == {
        "tf32_requested": "off",
        "tf32_effective": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    assert snapshot_tf32_execution_state() == caller
    with pytest.raises(ValueError, match="TF32 policy"):
        configure_tf32_execution(
            "invalid",
            real_dtype=torch.float32,
            device="cuda",
        )
    assert snapshot_tf32_execution_state() == caller


def test_scope_restores_exact_caller_state_after_success(restore_tf32_state):
    _set_state("medium", True, False)
    caller = snapshot_tf32_execution_state()

    with scoped_tf32_execution(
        "off",
        real_dtype=torch.float64,
        device="cpu",
    ) as metadata:
        assert metadata["tf32_requested"] == "off"
        assert metadata["tf32_effective"] is False
        assert snapshot_tf32_execution_state().to_metadata() == {
            "torch_float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }

    assert snapshot_tf32_execution_state() == caller


def test_scope_restores_exact_caller_state_after_exception(
    restore_tf32_state,
):
    _set_state("highest", False, True)
    caller = snapshot_tf32_execution_state()

    with pytest.raises(RuntimeError, match="owned failure"):
        with scoped_tf32_execution(
            "on",
            real_dtype=torch.float32,
            device="cuda",
        ):
            assert snapshot_tf32_execution_state().to_metadata() == {
                "torch_float32_matmul_precision": "high",
                "cuda_matmul_allow_tf32": True,
                "cudnn_allow_tf32": True,
            }
            raise RuntimeError("owned failure")

    assert snapshot_tf32_execution_state() == caller


def test_nested_scopes_restore_strictly_in_lifo_order(restore_tf32_state):
    _set_state("medium", True, False)
    caller = snapshot_tf32_execution_state()

    with scoped_tf32_execution(
        "off",
        real_dtype=torch.float32,
        device="cuda",
    ):
        outer = snapshot_tf32_execution_state()
        assert outer.to_metadata() == {
            "torch_float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        with scoped_tf32_execution(
            "on",
            real_dtype=torch.float32,
            device="cuda",
        ):
            assert snapshot_tf32_execution_state().to_metadata() == {
                "torch_float32_matmul_precision": "high",
                "cuda_matmul_allow_tf32": True,
                "cudnn_allow_tf32": True,
            }
        assert snapshot_tf32_execution_state() == outer

    assert snapshot_tf32_execution_state() == caller


def test_rc456_record_binds_exact_scope_sources_and_next_stage():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_RC4_5_6_SCOPED_TF32_CPU_CANDIDATE"
    )
    assert record["status"] == (
        "implementation_complete_cpu_verified_cuda_pending"
    )
    assert set(record["runtime_families"]) == {
        "plane_legacy_production",
        "plane_compiled_v2",
        "periodic_complete_stress",
        "channel_legacy",
        "channel_compiled_v2",
        "channel_complete_stress",
        "periodic_functional",
        "channel_functional",
    }
    for relative, expected in record["source_sha256"].items():
        resolved = _resolve_rc457_successor(relative, expected)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolved
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        resolved = _resolve_rc457_successor(
            relative, registration["current_sha256"]
        )
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolved
    assert record["scope"]["H100_executed"] is False
    assert record["authorization"]["rc4_5_8_h100_authorized"] is False
    assert record["authorization"][
        "eligible_for_rc4_5_7_installed_cpu_closure"
    ] is True

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    b11 = next(item for item in matrix["items"] if item["id"] == "B11")
    assert b11["state"] == "partial_residual"
    assert b11["historical_closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
    )
    progress = matrix["progress"]
    assert progress["rc4_5_6_complete"] is True
    assert progress["rc4_5_6_record_sha256"] == hashlib.sha256(
        RECORD.read_bytes()
    ).hexdigest()
    assert progress["rc4_5_8_h100_complete"] is True
