"""Executable RC4.5.3 checkpoint transaction and error contracts."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from pssolver.functional.errors import (
    FunctionalCheckpointCompatibilityError,
    FunctionalCheckpointError,
    FunctionalCheckpointIOError,
    translate_functional_exception,
)
from pssolver.io.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointIOError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
    checkpoint_directory_transaction,
    read_checkpoint_json,
    require_exact_checkpoint_keys,
    resolve_checkpoint_payload,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RC454 = ROOT / "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
RC455 = ROOT / "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"


@contextmanager
def _caller_umask(value: int):
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_generic_checkpoint_error_taxonomy_is_machine_readable():
    expected = {
        CheckpointNotFoundError: "checkpoint_not_found",
        CheckpointExistsError: "checkpoint_exists",
        CheckpointIntegrityError: "checkpoint_integrity_error",
        CheckpointCompatibilityError: "checkpoint_compatibility_error",
        CheckpointIOError: "checkpoint_io_error",
    }
    for error_type, code in expected.items():
        error = error_type(
            "failure",
            operation="checkpoint_import",
            failure_domain="filesystem",
            legacy_exception_type="OSError",
        )
        assert isinstance(error, CheckpointError)
        assert error.to_metadata() == {
            "code": code,
            "message": "failure",
            "operation": "checkpoint_import",
            "failure_domain": "filesystem",
            "legacy_exception_type": "OSError",
            "details": {},
        }


def test_functional_checkpoint_categories_retain_both_hierarchies():
    compatibility = FunctionalCheckpointCompatibilityError("incompatible")
    assert isinstance(compatibility, FunctionalCheckpointError)
    assert isinstance(compatibility, CheckpointCompatibilityError)

    cause = CheckpointIOError(
        "parent is a file",
        operation="channel_checkpoint_export",
        failure_domain="filesystem",
        legacy_exception_type="FileExistsError",
    )
    translated = translate_functional_exception(
        cause,
        operation="checkpoint_export",
    )
    assert isinstance(translated, FunctionalCheckpointIOError)
    assert isinstance(translated, CheckpointIOError)
    assert translated.details["legacy_exception_type"] == "FileExistsError"
    metadata = translated.to_metadata()
    assert metadata["failure_domain"] == "filesystem"
    assert metadata["legacy_exception_type"] == "FileExistsError"


def test_exact_schema_rejects_unknown_and_reports_missing_keys():
    with pytest.raises(CheckpointIntegrityError) as error:
        require_exact_checkpoint_keys(
            {"required": 1, "unknown": 2},
            required={"required", "missing"},
            description="test checkpoint",
            operation="checkpoint_import",
        )
    assert error.value.details == {
        "missing_keys": ["missing"],
        "unknown_keys": ["unknown"],
    }


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": 1, "value": 2}',
        '{"value": NaN}',
        '{"value": Infinity}',
        "[]",
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(tmp_path, payload):
    path = tmp_path / "checkpoint.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint_json(path, operation="checkpoint_import")


def test_strict_json_rejects_symlink_and_classifies_missing(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    linked = tmp_path / "checkpoint.json"
    linked.symlink_to(real)
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint_json(linked, operation="checkpoint_import")
    with pytest.raises(CheckpointNotFoundError):
        read_checkpoint_json(
            tmp_path / "missing.json",
            operation="checkpoint_import",
        )


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../payload.npy", "nested/payload.npy", "nested\\payload.npy"],
)
def test_payload_resolution_rejects_non_basename_paths(tmp_path, filename):
    with pytest.raises(CheckpointIntegrityError):
        resolve_checkpoint_payload(
            tmp_path,
            filename,
            operation="checkpoint_import",
            description="tensor payload",
        )


def test_payload_resolution_requires_canonical_regular_non_symlink(tmp_path):
    payload = tmp_path / "state.npy"
    payload.write_bytes(b"payload")
    assert resolve_checkpoint_payload(
        tmp_path,
        "state.npy",
        operation="checkpoint_import",
        description="tensor payload",
        expected_filename="state.npy",
    ) == payload.resolve()

    with pytest.raises(CheckpointIntegrityError):
        resolve_checkpoint_payload(
            tmp_path,
            "state.npy",
            operation="checkpoint_import",
            description="tensor payload",
            expected_filename="other.npy",
        )
    link = tmp_path / "linked.npy"
    link.symlink_to(payload)
    with pytest.raises(CheckpointIntegrityError):
        resolve_checkpoint_payload(
            tmp_path,
            "linked.npy",
            operation="checkpoint_import",
            description="tensor payload",
        )


@pytest.mark.parametrize("mask", [0o002, 0o022, 0o077])
def test_writer_transaction_respects_caller_umask(tmp_path, mask):
    target = tmp_path / f"checkpoint-{mask:o}"
    with _caller_umask(mask):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, published):
            assert published == target
            with (staging / "payload.bin").open("xb") as handle:
                handle.write(b"payload")
            (staging / "checkpoint.json").write_text(
                json.dumps({"complete": True}) + "\n",
                encoding="utf-8",
            )
    assert _mode(target) == 0o777 & ~mask
    assert _mode(target / "payload.bin") == 0o666 & ~mask
    assert _mode(target / "checkpoint.json") == 0o666 & ~mask
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))


def test_writer_transaction_refuses_existing_target_without_mutation(tmp_path):
    target = tmp_path / "checkpoint"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(CheckpointExistsError):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ):
            raise AssertionError("body must not run")
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_writer_transaction_classifies_parent_shape_and_preserves_cause(tmp_path):
    parent = tmp_path / "parent"
    parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(CheckpointIOError) as error:
        with checkpoint_directory_transaction(
            parent / "checkpoint",
            operation="checkpoint_export",
        ):
            pass
    assert isinstance(error.value.__cause__, OSError)
    assert error.value.legacy_exception_type in {
        "FileExistsError",
        "NotADirectoryError",
    }


@pytest.mark.parametrize(
    "failure_point",
    ["first_payload", "middle_payload", "metadata", "origin_marker"],
)
def test_writer_transaction_removes_failed_body_staging(
    tmp_path,
    failure_point,
):
    target = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError, match="injected payload failure"):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            (staging / f"{failure_point}.partial").write_bytes(b"partial")
            raise RuntimeError("injected payload failure")
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_writer_transaction_translates_body_oserror_and_preserves_cause(tmp_path):
    target = tmp_path / "checkpoint"
    with pytest.raises(CheckpointIOError) as error:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ):
            raise PermissionError(errno.EACCES, "injected permission failure")
    assert isinstance(error.value.__cause__, PermissionError)
    assert error.value.legacy_exception_type == "PermissionError"
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_writer_transaction_removes_staging_after_publish_failure(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "checkpoint"

    def fail_publish(_source, _target):
        raise OSError(errno.ENOSPC, "injected device-full failure")

    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises(CheckpointIOError) as error:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            (staging / "payload.bin").write_bytes(b"complete")
    assert isinstance(error.value.__cause__, OSError)
    assert error.value.legacy_exception_type == "OSError"
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_rc453_record_binds_contract_sources_and_scope():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    contract = ROOT / record["baseline"]["contract_record"]
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == record[
        "baseline"
    ]["contract_record_sha256"]
    assert record["classification"] == (
        "PASS_RC4_5_3_CHECKPOINT_TRANSACTIONALITY_CPU_CANDIDATE"
    )
    assert record["issues"] == {
        "B7": "closed_verified",
        "B24": "closed_verified",
        "B27": "closed_verified",
    }
    for relative, expected in record["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual == expected:
            continue
        resolved = expected
        for successor_path in (RC454, RC455):
            registration = json.loads(
                successor_path.read_text(encoding="utf-8")
            )["historical_successor_sources"].get(relative)
            if registration is None:
                continue
            if registration["historical_sha256"] == resolved:
                resolved = registration["current_sha256"]
        assert resolved == actual
    parent = record["baseline"]["implementation_parent_commit"]
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]
        resolved = registration["current_sha256"]
        successor = json.loads(RC455.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolved
    assert record["scope"] == {
        "checkpoint_schema_or_tensor_layout_version_changed": False,
        "scientific_model_changed": False,
        "numerical_hot_path_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "h100_executed": False,
        "push_executed": False,
    }


def test_rc453_record_freezes_transaction_and_next_stage_boundary():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert set(record["checkpoint_surfaces"]) == {
        "plane_production",
        "legacy_channel_production",
        "periodic_production",
        "channel_complete_stress_production",
        "finite_q_anchoring",
        "periodic_functional",
        "channel_functional",
    }
    assert record["writer_transaction"] == {
        "same_filesystem_sibling_staging": True,
        "single_atomic_publish": True,
        "target_overwrite_allowed": False,
        "failed_target_or_staging_residue_allowed": False,
        "caller_umask_respected": True,
        "umask_matrix": ["002", "022", "077"],
    }
    assert record["authorization"] == {
        "rc4_5_3_implementation_authorized": True,
        "rc4_5_3_complete": True,
        "eligible_for_rc4_5_4_implementation": True,
        "rc4_5_4_implementation_authorized": False,
        "PSSolver_Control_modification_authorized": False,
        "h100_submission_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def test_known_issue_matrix_preserves_rc453_and_registers_rc454_successor():
    progress = json.loads(MATRIX.read_text(encoding="utf-8"))["progress"]
    record_hash = hashlib.sha256(RECORD.read_bytes()).hexdigest()
    assert progress["rc4_5_3_implementation_authorized"] is True
    assert progress["rc4_5_3_complete"] is True
    assert progress["rc4_5_3_classification"] == (
        "PASS_RC4_5_3_CHECKPOINT_TRANSACTIONALITY_CPU_CANDIDATE"
    )
    assert progress["rc4_5_3_record"] == RECORD.relative_to(ROOT).as_posix()
    assert progress["rc4_5_3_record_sha256"] == record_hash
    assert progress["eligible_for_rc4_5_4_implementation"] is True
    assert progress["rc4_5_4_implementation_authorized"] is True
    assert progress["rc4_5_4_complete"] is True
    assert progress["eligible_for_rc4_5_5_implementation"] is True
    assert progress["rc4_5_5_implementation_authorized"] is True
    assert progress["rc4_5_5_complete"] is True
    assert progress["eligible_for_rc4_5_6_implementation"] is True
