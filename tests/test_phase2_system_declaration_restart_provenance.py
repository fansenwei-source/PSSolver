"""Golden restart-provenance gate across the P2.1S declaration extraction."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASES = json.loads(
    (PROJECT_ROOT / "tests" / "fixtures" / "systems" / "algebraic_stokes_cases_v1.json").read_text(
        encoding="utf-8"
    )
)
