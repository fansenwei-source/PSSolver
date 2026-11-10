"""References remain exact, portable repository artifacts with strict gates."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pssolver.io.checkpoint import CheckpointError, seal_checkpoint_metadata
from qualification.identity.checkpoint_fixtures import file_manifest
from qualification.identity.records import require_fingerprint
from qualification.identity.runtime_case_v2 import RuntimeCaseV2 as RuntimeCase


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/v0_4_0_s23"


def test_wrong_fingerprint_fails_without_skip_or_numeric_tolerance():
    with pytest.raises(
        RuntimeError, match="PSSOLVER_TEST_PREREQUISITE: baseline fingerprint differs"
    ):
        require_fingerprint({"torch": "changed"}, {"torch": "baseline"})


def test_authentic_release_fixture_tree_is_small_and_completely_tracked():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    actual = file_manifest(FIXTURES)
    del actual["manifest.json"]
    assert actual == manifest["files"]
    assert sum(row["size"] for row in actual.values()) < 5_000_000
    assert manifest["writer_version"] == "0.4.0"
    assert (
        manifest["wheel_sha256"]
        == "c5cf3e26af8bdde27287f6129fe5939d6f202fb627dcdc9a66c19d62871dd873"
    )
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "tests/fixtures/v0_4_0_s23"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    )
    assert {"tests/fixtures/v0_4_0_s23/" + name for name in [*actual, "manifest.json"]} <= tracked


@pytest.mark.parametrize(
    "scene",
    json.loads((FIXTURES / "manifest.json").read_text())["scenes"],
    ids=lambda row: row["id"],
)
def test_unknown_format_is_rejected_before_any_tensor_read(scene, tmp_path, monkeypatch):
    work = tmp_path / "runtime"
    work.mkdir()
    case = RuntimeCase(scene, work)
    metadata = json.loads((FIXTURES / scene["id"] / "checkpoint.json").read_text())
    metadata["format_version"] = 999
    metadata.pop("metadata_sha256", None)
    metadata = seal_checkpoint_metadata(metadata)
    invalid = tmp_path / "unsupported"
    invalid.mkdir()
    (invalid / "checkpoint.json").write_text(json.dumps(metadata))
    attempts = []

    def forbidden_read(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("Tensor loading occurred before format rejection")

    monkeypatch.setattr(np, "load", forbidden_read)
    with pytest.raises(
        (CheckpointError, ValueError),
        match="(?i)format|version|^unsupported complete-stress Channel checkpoint$",
    ):
        case.restore(invalid, 2)
    assert attempts == []
