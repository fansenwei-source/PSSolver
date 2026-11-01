"""Bindings for the local S-18 R4 step/progress candidate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s18_r4_step_progress.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_r4_candidate_binds_confirmed_design_and_exact_runtime_source():
    record = _read()
    assert record["plan_step"] == "S-18"
    assert record["batch"] == "R4"
    assert record["classification"] == (
        "R4_LOCAL_CPU_CANDIDATE_AWAITING_INDEPENDENT_REVIEW"
    )
    assert record["governance"] == {
        "decision": "D-0018",
        "design_review": "R-0054",
        "implementation_commit": (
            "9cf54553b9f1bb2b94c2807c2e47adedeb415846"
        ),
        "implementation_parent": (
            "9ffe2fe9d148e10bc5ec6c23cdac373e5ebdb280"
        ),
        "comparison_baseline": "v0.3.0",
        "nonzero_tolerance_authorized": False,
    }


def test_r4_candidate_requires_every_pre_registered_cpu_gate():
    record = _read()
    assert all(record["required_evidence"].values())
    assert record["s19_handoff"] == [
        "compiled-v2 torch.compile graph and graph-break closure",
        "CUDA transform order and counters",
        "CUDA restart and long-running finite-state checks",
        "H100 timing and memory non-regression",
    ]


def test_r4_candidate_changes_no_public_contract_or_authorization():
    record = _read()
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_implementation": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
