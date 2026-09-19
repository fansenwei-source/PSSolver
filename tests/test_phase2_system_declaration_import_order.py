"""Fresh-process import-order gates for the P2.1S declaration baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "systems"
    / "algebraic_stokes_cases_v1.json"
)
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
LEGACY_PICKLE_BASE64 = CASES["legacy_stokes_pickle"]["base64"]
LEGACY_ALGEBRAIC_CLASS_BASE64 = CASES[
    "legacy_algebraic_update_phase_pickle"
]["class_base64"]
LEGACY_ALGEBRAIC_MEMBER_BASE64 = CASES[
    "legacy_algebraic_update_phase_pickle"
]["member_base64"]


IMPORT_ORDERS = {
    "leaf_first": """
import pssolver.execution.algebraic as algebraic
import pssolver.execution.stokes as stokes
import pssolver.execution as execution
assert algebraic.AlgebraicSystemSpec is execution.AlgebraicSystemSpec
assert algebraic.AlgebraicUpdatePhase is execution.AlgebraicUpdatePhase
assert stokes.IncompressibleStokesSystemSpec is execution.IncompressibleStokesSystemSpec
assert stokes.PressureGauge is execution.PressureGauge
assert stokes.TangentialZeroModePolicy is execution.TangentialZeroModePolicy
""",
    "package_first": """
import pssolver.execution as execution
import pssolver.execution.algebraic as algebraic
import pssolver.execution.stokes as stokes
assert execution.AlgebraicSystemSpec is algebraic.AlgebraicSystemSpec
assert execution.IncompressibleStokesSystemSpec is stokes.IncompressibleStokesSystemSpec
assert execution.PressureGauge.ZERO_MEAN is stokes.PressureGauge.ZERO_MEAN
assert execution.TangentialZeroModePolicy.FRICTION is stokes.TangentialZeroModePolicy.FRICTION
""",
    "model_first": """
import pssolver.models.active_nematics.constitutive as constitutive
import pssolver.execution as execution
import pssolver.execution.stokes as stokes
assert constitutive.IncompressibleStokesSystemSpec is execution.IncompressibleStokesSystemSpec
assert execution.IncompressibleStokesSystemSpec is stokes.IncompressibleStokesSystemSpec
""",
    "legacy_pickle_first": f"""
import base64
import pickle
payload = base64.b64decode({LEGACY_PICKLE_BASE64!r})
value = pickle.loads(payload)
import pssolver.execution as execution
assert type(value) is execution.IncompressibleStokesSystemSpec
assert value.pressure_gauge is execution.PressureGauge.ZERO_MEAN
assert value.tangential_zero_mode_policy is execution.TangentialZeroModePolicy.ZERO_MEAN
""",
    "legacy_algebraic_pickle_first": f"""
import base64
import pickle
enum_type = pickle.loads(base64.b64decode({LEGACY_ALGEBRAIC_CLASS_BASE64!r}))
member = pickle.loads(base64.b64decode({LEGACY_ALGEBRAIC_MEMBER_BASE64!r}))
import pssolver.execution as execution
import pssolver.execution.algebraic as algebraic
assert enum_type is execution.AlgebraicUpdatePhase
assert member is execution.AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
assert algebraic.AlgebraicUpdatePhase is execution.AlgebraicUpdatePhase
assert algebraic.AlgebraicSystemSpec is execution.AlgebraicSystemSpec
""",
}


@pytest.mark.parametrize(
    "case",
    sorted(IMPORT_ORDERS),
)
def test_declaration_import_orders_are_cycle_free_and_identity_stable(case):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_ORDERS[case]],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
