"""S-07 packaging, provenance, observability, and documentation contracts."""

from __future__ import annotations

import ast
import dataclasses
import errno
import importlib.util
import logging
from pathlib import Path

import pytest

import pssolver.experimental as experimental
import pssolver.io.checkpoint as checkpoint_io
from pssolver.control.active_force import (
    active_force_divergence as compatibility_active_force_divergence,
)
from pssolver.functional.channel_checkpoint import (
    _integrity_status as channel_integrity_status,
)
from pssolver.functional.contracts import FunctionalCheckpointState
from pssolver.functional.periodic_checkpoint import (
    _integrity_status as periodic_integrity_status,
)
from pssolver.io import CheckpointPublicationDiagnostic
from pssolver.models.active_nematics import active_force_divergence
from pssolver.snapshots import _continuation_signature


ROOT = Path(__file__).resolve().parents[1]


def _force_nfs_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported(_source, _target, *, operation):
        raise checkpoint_io._RenameNoReplaceUnsupported(
            "injected NFS capability gap",
            operation=operation,
            failure_domain="filesystem",
            details={"errno": errno.EINVAL, "filesystem": "nfs"},
        )

    monkeypatch.setattr(
        checkpoint_io,
        "_rename_directory_no_replace",
        unsupported,
    )


@pytest.mark.parametrize(
    ("force_fallback", "expected_backend"),
    (
        (False, "rename_noreplace"),
        (True, "owned_placeholder_nfs_compatibility"),
    ),
)
def test_checkpoint_publication_backend_is_observable_but_not_identity(
    tmp_path,
    monkeypatch,
    caplog,
    force_fallback,
    expected_backend,
):
    if force_fallback:
        _force_nfs_fallback(monkeypatch)
    target = tmp_path / "checkpoint"

    with caplog.at_level(logging.INFO, logger="pssolver.io.checkpoint"):
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, published):
            assert published == target
            (staging / "payload.bin").write_bytes(b"payload")

    records = [
        record
        for record in caplog.records
        if hasattr(record, "checkpoint_publication")
    ]
    assert len(records) == 1
    diagnostic = records[0].checkpoint_publication
    assert diagnostic == {
        "schema_version": 1,
        "operation": "checkpoint_export",
        "target": str(target),
        "backend": expected_backend,
        "durability_confirmed": True,
        "checkpoint_identity_member": False,
    }
    assert (target / "payload.bin").read_bytes() == b"payload"
    assert sorted(path.name for path in target.iterdir()) == ["payload.bin"]


def test_public_checkpoint_diagnostic_is_frozen_and_explicitly_nonidentity():
    diagnostic = CheckpointPublicationDiagnostic(
        operation="checkpoint_export",
        target="checkpoint",
        backend="rename_noreplace",
    )
    assert diagnostic.to_metadata()["checkpoint_identity_member"] is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        diagnostic.backend = "other"


def test_publication_diagnostic_failure_cannot_mask_durable_success(
    tmp_path,
):
    class RaisingFilter(logging.Filter):
        def filter(self, record):
            raise RuntimeError("user logging filter failed")

    logger = logging.getLogger("pssolver.io.checkpoint")
    filter_ = RaisingFilter()
    logger.addFilter(filter_)
    target = tmp_path / "checkpoint"
    try:
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, published):
            assert published == target
            (staging / "payload.bin").write_bytes(b"durable")
    finally:
        logger.removeFilter(filter_)

    assert (target / "payload.bin").read_bytes() == b"durable"


def test_legacy_active_force_is_a_forward_to_the_packaged_implementation():
    assert compatibility_active_force_divergence is active_force_divergence


def test_functional_checkpoint_import_reports_sealed_or_legacy_integrity():
    sealed = {checkpoint_io.METADATA_SHA256_KEY: "0" * 64}
    unsealed = {"format_version": 1}
    assert channel_integrity_status(sealed) == "sealed_sha256"
    assert periodic_integrity_status(sealed) == "sealed_sha256"
    assert channel_integrity_status(unsealed) == "unsealed_legacy"
    assert periodic_integrity_status(unsealed) == "unsealed_legacy"
    assert "integrity_status" in {
        field.name for field in dataclasses.fields(FunctionalCheckpointState)
    }


def test_plane_v2_application_identity_matches_legacy_resume_identity():
    common = {
        "solver": {"shape": [8, 8, 8], "lengths": [1, 1, 1], "dt": 0.1},
        "model": {
            "name": "active_nematics",
            "Q_convention": {"representation": "traceless_symmetric_3x3"},
            "parameters": {"activity": 0.01},
        },
        "boundary_conditions": {"Q": "neumann"},
        "numerics": {"dtype": "float64"},
    }
    legacy = {
        **common,
        "schema_version": 1,
        "script": "Plane_beris_edwards_stokes.py",
    }
    current = {
        **common,
        "schema_version": 2,
        "application": "plane_complete_stress_beris_edwards",
    }
    assert _continuation_signature(legacy) == _continuation_signature(current)
    assert "script" not in _continuation_signature(current)


def test_wheel_policy_excludes_legacy_control_and_qualification_modules():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'exclude=["pssolver.control", "pssolver.control.*"]' in setup
    assert 'item[1].startswith("stage_")' in setup
    assert 'item[1] != "h100_shadow_qualification"' in setup
    assert not any(name.startswith("Stage") for name in experimental.__all__)
    assert not any("Qualification" in name for name in experimental.__all__)


def test_every_retained_wheel_source_is_closed_over_excluded_control_package():
    violations = []
    package_root = ROOT / "pssolver"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("pssolver", "control"):
            continue
        if relative.parts[:2] == ("pssolver", "experimental") and (
            path.stem.startswith("stage_")
            or path.stem == "h100_shadow_qualification"
        ):
            continue
        module_parts = list(relative.with_suffix("").parts)
        is_package = module_parts[-1] == "__init__"
        if is_package:
            module_parts.pop()
        module = ".".join(module_parts)
        package = module if is_package else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pssolver.control" or alias.name.startswith(
                        "pssolver.control."
                    ):
                        violations.append((str(relative), node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    imported = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""),
                        package,
                    )
                else:
                    imported = node.module
                if imported == "pssolver.control" or (
                    imported is not None
                    and imported.startswith("pssolver.control.")
                ):
                    violations.append((str(relative), node.lineno, imported))
    assert violations == []


def test_robin_and_sealed_integrity_boundaries_are_documented():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scope = (ROOT / "notes/pssolver_v0_2_scope.md").read_text(
        encoding="utf-8"
    )
    assert "independently reported boundary" in changelog
    assert "not a MAC or digital signature" in changelog
    assert 'integrity_status == "sealed_sha256"' in changelog
    assert "independent boundary-contract gate" in readme
    assert "separate interior Helmholtz" in " ".join(scope.split())
