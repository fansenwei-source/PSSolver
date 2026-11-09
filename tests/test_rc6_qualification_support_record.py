"""Bind the RC6 qualification-support record to the implemented files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_qualification_support.json"
SUCCESSOR_RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_nvcc_bootstrap_recovery.json"
PROFILE_TF32_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_profile_child_tf32_recovery.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
