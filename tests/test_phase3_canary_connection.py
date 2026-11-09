"""P3.4 local gates for the separated-canary state connection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


def _production_reference(tmp_path: Path) -> tuple[dict[str, object], dict]:
    output = tmp_path / "legacy_reference"
    result = subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCRIPT),
            "--activity-number",
            "18",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--pointwise-execution",
            "eager",
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--steps",
            "1",
            "--save-start-step",
            "0",
            "--save-interval",
            "1",
            "--spectral-refresh-steps",
            "2",
            "--save-hydrodynamics",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    q = np.load(output / "Q_0.npy", allow_pickle=False)
    initial = {
        name: torch.from_numpy(np.array(q[..., index], copy=True))
        for index, name in enumerate(Q_COMPONENTS)
    }
    return metadata, initial


def _assert_all_field_storage_equal(left, right):
    assert torch.equal(left.solver.fields.spatial, right.solver.fields.spatial)
    assert torch.equal(left.solver.fields.spectral, right.solver.fields.spectral)
