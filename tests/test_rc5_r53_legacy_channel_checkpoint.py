"""R5-3 regression tests for released P7 legacy Channel checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


FIXTURES = Path(__file__).parent / "fixtures" / "legacy_channel_checkpoints_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(directory: Path) -> tuple[int, str]:
    files = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    payload = json.dumps(
        files,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return len(files), hashlib.sha256(payload).hexdigest()


def _state(adapter) -> dict[str, object]:
    return {
        "spatial": adapter.fields.spatial.detach().clone(),
        "spectral": adapter.fields.spectral.detach().clone(),
        "pressure": adapter.capture_pressure_guess(),
        "completed_steps": adapter.completed_steps,
        "integrator_step_count": int(adapter.solver.integrator.step_count),
        "integrator_refresh_count": int(adapter.solver.integrator.refresh_count),
    }


def _assert_state(adapter, expected: dict[str, object]) -> None:
    assert torch.equal(adapter.fields.spatial, expected["spatial"])
    assert torch.equal(adapter.fields.spectral, expected["spectral"])
    assert torch.equal(adapter.capture_pressure_guess(), expected["pressure"])
    assert adapter.completed_steps == expected["completed_steps"]
    assert int(adapter.solver.integrator.step_count) == expected[
        "integrator_step_count"
    ]
    assert int(adapter.solver.integrator.refresh_count) == expected[
        "integrator_refresh_count"
    ]


def test_released_fixture_manifest_is_provenance_bound():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["fixture_schema_version"] == 1
    assert set(manifest["cases"]) == {
        "rc1_even",
        "rc1_odd",
        "rc2_even",
        "rc3_even",
    }
    for name, record in manifest["cases"].items():
        directory = FIXTURES / name
        assert _sha256(directory / "checkpoint.json") == record[
            "checkpoint_sha256"
        ]
        assert _tree_sha256(directory) == (
            record["file_count"],
            record["tree_sha256"],
        )
