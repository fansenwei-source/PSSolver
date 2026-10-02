"""Machine-readable P9.7.4 record consistency."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver
import pssolver.functional as functional


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/architecture_v0_2/phase_9_p974_channel_functional_runtime.json"
FROZEN_SOLVER = ROOT / "pssolver/linear_solvers/stokes/channel_no_slip.py"


def test_p974_record_is_json_safe_and_keeps_frozen_solver_byte_identity():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["stage"] == "P9.7.4"
    assert record["classification"] == "PASS_P9_7_4_CHANNEL_FUNCTIONAL_RUNTIME"
    assert record["capabilities"]["pure_step"] is True
    assert record["capabilities"]["deterministic_replay"] == "bitwise"
    assert record["capabilities"]["checkpoint_bridge"] == (
        "functional_q_state_only_with_production_q_import"
    )
    assert record["pressure_policy"]["functional_warm_start"] is False
    assert record["pressure_policy"]["custom_implicit_adjoint"] is True
    assert FROZEN_SOLVER.is_file()
    # P9.7.4 records the solver that was qualified at that phase.  It remains
    # immutable historical evidence after the rc2 even-grid Nyquist repair.
    assert record["frozen_solver"]["sha256"] == (
        "4503f62df5204f7f48da7ca58674ebda45a3f775b7742dcc7bc7ab535c511b16"
    )
    assert record["boundaries"]["p9_7_5_implemented"] is False
    assert record["boundaries"]["h100_job_submitted"] is False
    assert record["boundaries"]["control_repository_modified"] is False
    assert hasattr(functional, "ChannelActivityFunctionalRuntime")
    assert hasattr(functional, "build_channel_activity_functional_runtime")
    assert not hasattr(pssolver, "ChannelActivityFunctionalRuntime")

    archive = (
        ROOT / "notes/architecture_v0_2/build_verbatim_archive_pdf.py"
    ).read_text(encoding="utf-8")
    markdown = '"phase_9_p974_channel_functional_runtime.md"'
    machine = '"phase_9_p974_channel_functional_runtime.json"'
    assert archive.count(markdown) == 1
    assert archive.count(machine) == 1
    assert archive.index(markdown) < archive.index(machine)
