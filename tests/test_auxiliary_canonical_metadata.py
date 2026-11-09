import argparse
import json

import numpy as np

from pssolver.models.active_nematics import (
    Q_convention_metadata,
    positive_equilibrium_S,
)


S_BULK = positive_equilibrium_S(-1.0, -6.0, 6.0)


S_INITIAL = 2.0 / 3.0


SHAPE = (8, 6, 6)


def write_canonical_q(
    directory,
    *,
    actual_S=S_INITIAL,
    S_initial=None,
    S_bulk=S_BULK,
    convention=None,
):
    directory.mkdir()
    q = np.zeros((*SHAPE, 5), dtype=np.float32)
    q[..., 0] = actual_S
    q[..., 3] = -0.5 * actual_S
    q_path = directory / "Q.npy"
    np.save(q_path, q)
    metadata = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": (
                Q_convention_metadata() if convention is None else convention
            ),
            "parameters": {
                "S_initial": actual_S if S_initial is None else S_initial,
                "S_bulk": S_bulk,
            },
        }
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return q_path, q


def validation_args(tmp_path):
    initial_path, _ = write_canonical_q(tmp_path / "initial")
    target_path, _ = write_canonical_q(tmp_path / "target")
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.ones(SHAPE, dtype=np.float32))
    return argparse.Namespace(
        initial=initial_path,
        target=target_path,
        mask=mask_path,
        stride=(1, 1, 1),
        steps=1,
        dt=1.0e-4,
        alpha_mean=0.0,
        alpha_amplitude=0.0,
        auto_mask_margin=(0, 0, 0),
        auto_mask_relative_threshold=0.2,
        mask_transition_width=1.5,
        pressure_rel_tol=1.0e-6,
        pressure_max_iter=30,
        rtol=1.0e-4,
        atol=1.0e-5,
        report_steps=(),
        device="cpu",
        output=tmp_path / "output",
    )
