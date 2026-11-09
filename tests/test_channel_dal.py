import json

import numpy as np

from pssolver import write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
)


def _write_Q_input(
    directory,
    *,
    actual_S=0.4,
    declared_S=None,
    model_name="active_nematics",
    convention=None,
):
    directory.mkdir()
    Q = np.zeros((6, 4, 3, 5), dtype=np.float32)
    Q[..., 0] = actual_S
    Q[..., 3] = -actual_S / 2.0
    Q_path = directory / "Q.npy"
    np.save(Q_path, Q)
    metadata = {
        "schema_version": 1,
        "model": {
            "name": model_name,
            "Q_convention": (Q_convention_metadata() if convention is None else convention),
            "parameters": {
                "S_initial": actual_S if declared_S is None else declared_S,
                "S_bulk": actual_S if declared_S is None else declared_S,
            },
        },
        "classification": {
            "type": "pure_splay_loop",
            "classification_regime": "ordered",
        },
        "loop": {"center": [3.0, 2.0, 1.5], "radius": 1.0},
        "validation": {
            "defect_plaquette_counts": {"xy": 0, "xz": 8, "yz": 8},
            "yz_plane_topology": True,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return Q_path


def test_run_metadata_status_updates_are_atomic_and_json_serializable(tmp_path):
    payload = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {"S_initial": 0.4, "S_bulk": 0.4},
        },
    }

    metadata_path = write_run_metadata(tmp_path, payload, status="running")
    assert json.loads(metadata_path.read_text())["status"] == "running"
    assert not (tmp_path / "metadata.json.tmp").exists()

    write_run_metadata(tmp_path, payload, status="complete")
    complete = json.loads(metadata_path.read_text())
    assert complete["status"] == "complete"
    assert complete["model"] == payload["model"]
    assert not (tmp_path / "metadata.json.tmp").exists()
