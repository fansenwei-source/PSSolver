"""Test-only old-name loader for the seven independent frozen stage plans.

Install explicitly from tests/conftest.py or qualification/run.py. Production
entry points and distributed packages never install this finder. B-group stage
audits run in a separate, fixed v0.4.0 checkout instead.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path


ALLOWED = frozenset(
    {
        "stage_m_plan",
        "stage_n_plan",
        "stage_n1_plan",
        "stage_n2_plan",
        "stage_n3_plan",
        "stage_n4_plan",
        "stage_o4_plan",
    }
)
PREFIX = "pssolver.experimental."
STAGES = Path(__file__).resolve().parent / "stages"


class StagePlanFinder(importlib.abc.MetaPathFinder):
    """Resolve only the frozen A-group names, retaining their module identity."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(PREFIX) and fullname[len(PREFIX) :] in ALLOWED:
            return importlib.util.spec_from_file_location(
                fullname, STAGES / (fullname[len(PREFIX) :] + ".py")
            )
        return None


def install() -> None:
    """Idempotently install the allowlisted finder in this test process."""
    if not any(isinstance(finder, StagePlanFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, StagePlanFinder())
