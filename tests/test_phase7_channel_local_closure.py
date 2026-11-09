"""P7.5 package application, output, and same-runtime restart closure."""

from __future__ import annotations
from pathlib import Path

import numpy as np


SHAPE = (8, 6, 5)


LENGTHS = (8.0, 6.0, 5.0)


def _load_outputs(directory: Path, step: int):
    return tuple(np.load(directory / f"{name}_{step}.npy", allow_pickle=False) for name in ("Q", "u", "p"))
