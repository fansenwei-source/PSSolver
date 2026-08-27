import json

import numpy as np
import pytest
import torch

from pssolver.control import load_q_target
from pssolver.models.active_nematics import Q_convention_metadata


def test_load_q_target_requires_colocated_canonical_metadata(tmp_path):
    q_path = tmp_path / "Q_10.npy"
    np.save(q_path, np.zeros((4, 3, 2, 5), dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="Canonical active-nematic Q metadata"):
        load_q_target(q_path)


def test_load_q_target_returns_component_first_tensor_after_validation(tmp_path):
    values = np.arange(4 * 3 * 2 * 5, dtype=np.float32).reshape(4, 3, 2, 5)
    q_path = tmp_path / "Q_10.npy"
    np.save(q_path, values)
    metadata = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "S_initial": 0.4,
                "S_bulk": 0.4,
            },
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    target = load_q_target(q_path, dtype=torch.float64)

    assert target.shape == (5, 1, 4, 3, 2)
    assert target.dtype == torch.float64
    np.testing.assert_allclose(
        target[:, 0].movedim(0, -1).numpy(),
        values,
    )
