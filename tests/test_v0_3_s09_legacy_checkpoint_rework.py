"""S-09 regressions for strict legacy API labels and v0.2.x migrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from pssolver.functional import FunctionalCheckpointCompatibilityError
from pssolver.io.checkpoint import (
    FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE,
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)
import pssolver.functional.periodic_checkpoint as periodic_checkpoint_module
from tests.test_rc5_legacy_functional_checkpoints import (
    _FIXTURE_ROOT,
    _channel_runtime,
    _periodic_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/s09_legacy_compatibility_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(directory: Path) -> str:
    files = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    encoded = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relabel_api(source: Path, target: Path, api_version: str) -> Path:
    shutil.copytree(source, target)
    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["functional_bridge"]["api_version"] = api_version
    if "metadata_sha256" in metadata:
        metadata.pop("metadata_sha256")
        metadata = seal_checkpoint_metadata(metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_path = target / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE
    if provenance_path.exists():
        provenance_path.unlink()
        write_functional_checkpoint_provenance(
            target,
            metadata,
            metadata["functional_bridge"],
        )
    return target


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    root = tmp_path_factory.mktemp("s09_legacy_checkpoint_rework")
    return {
        "periodic": _periodic_runtime(root / "periodic"),
        "channel": _channel_runtime(root / "channel"),
    }


@pytest.mark.parametrize("generation", ["rc1", "rc2", "rc3"])
@pytest.mark.parametrize("family", ["periodic", "channel"])
@pytest.mark.parametrize("api_version", ["1.1", "0.1-provisional", "9.9"])
def test_legacy_migration_rejects_non_1_0_api_before_tensor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtimes,
    generation: str,
    family: str,
    api_version: str,
) -> None:
    source = _relabel_api(
        _FIXTURE_ROOT / generation / f"{family}_func_ck",
        tmp_path / "source",
        api_version,
    )
    runtime = runtimes[family]

    def forbidden_payload_read(*_args, **_kwargs):
        raise AssertionError("tensor payload was read before API validation")

    if family == "channel":
        monkeypatch.setattr(
            runtime.checkpoint_bridge,
            "_load_tensor",
            forbidden_payload_read,
        )
    else:
        monkeypatch.setattr(
            periodic_checkpoint_module,
            "load_periodic_checkpoint",
            forbidden_payload_read,
        )
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="API version must be 1.0",
    ):
        runtime.checkpoint_bridge.migrate_legacy_checkpoint(
            source,
            tmp_path / "target",
            source_release_generation=generation,
        )
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("api_version", ["1.1", "0.1-provisional", "9.9"])
def test_periodic_v2_bridge_rejects_non_1_0_api_before_tensor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtimes,
    api_version: str,
) -> None:
    source = _relabel_api(
        _FIXTURE_ROOT / "rc2/periodic_func_ck",
        tmp_path / "source",
        api_version,
    )

    def forbidden_payload_read(*_args, **_kwargs):
        raise AssertionError("tensor payload was read before API validation")

    monkeypatch.setattr(
        periodic_checkpoint_module,
        "load_periodic_checkpoint",
        forbidden_payload_read,
    )
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="API version must be 1.0",
    ):
        runtimes["periodic"].checkpoint_bridge.import_checkpoint(source)


def test_v022_migrated_rc1_even_channel_fixture_is_authentic_and_readable(
    runtimes,
) -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    fixture = FIXTURE_ROOT / manifest["fixture"]["path"]
    assert manifest["source_runtime"]["commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )
    assert _tree_sha256(fixture) == manifest["fixture"]["tree_sha256"]
    for relative, digest in manifest["fixture"]["files"].items():
        assert _sha256(fixture / relative) == digest

    metadata = json.loads((fixture / "checkpoint.json").read_text())
    migration = metadata["migration_provenance"]
    assert migration == {
        "derivative_dynamics_change": "pre_rc3_pressure_transpose_nyquist",
        "migration_kind": "state_only_migration",
        "periodic_axis_size": 6,
        "source_api_version": "1.0",
        "source_family": "channel_functional",
        "source_format_version": 1,
        "source_release_generation": "rc1",
        "trajectory_equivalent": False,
    }
    imported = runtimes["channel"].checkpoint_bridge.import_checkpoint(fixture)
    assert imported.completed_steps == 2
    assert imported.compatibility is not None
    assert imported.compatibility.source_api_version == "1.0"
    assert imported.compatibility.target_api_version == "1.1"
    assert imported.compatibility.exact_current_protocol is False
    expected = (
        torch.from_numpy(np.load(fixture / "state__q_physical.npy")),
        torch.from_numpy(np.load(fixture / "state__q_spectral.npy")),
    )
    assert all(
        torch.equal(actual.cpu(), wanted)
        for actual, wanted in zip(imported.state, expected, strict=True)
    )
    controls = {
        "activity": torch.full(
            runtimes["channel"].control_specs[0].tensor.shape,
            0.01,
            dtype=torch.float64,
        )
    }
    stepped = runtimes["channel"].step(imported.state, controls, 2)
    assert all(bool(torch.isfinite(value).all()) for value in stepped)


def test_s09_fixture_manifest_files_are_all_tracked_by_git() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())
    fixture = FIXTURE_ROOT / manifest["fixture"]["path"]
    tracked = {
        line
        for line in subprocess.run(
            ["git", "ls-files", "--", str(FIXTURE_ROOT.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if line
    }
    expected = {
        (fixture / relative).relative_to(ROOT).as_posix()
        for relative in manifest["fixture"]["files"]
    }
    assert expected <= tracked
