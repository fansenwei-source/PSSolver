"""Static tests for the frozen P9.5 qualification plan."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN = NOTES / "phase_9_p95_h100_qualification_plan.json"


def test_p95_plan_freezes_batch_one_matrix_and_nonclaims():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["phase"] == "P9.5"
    assert plan["scope"]["batch_sizes_qualified"] == [1]
    assert plan["scope"]["larger_batch_qualified"] is False
    assert plan["profile_count"] == 18
    assert [grid["id"] for grid in plan["grids"]] == ["R128", "R320"]
    assert plan["roles"] == [
        "production_forward",
        "functional_forward",
        "functional_vjp",
    ]
    assert plan["h100_execution"]["submission_count_max"] == 1
    assert plan["h100_execution"]["automatic_retry"] is False
    assert plan["authorization"]["p9_6_authorized"] is False


def test_p95_plan_lists_every_existing_cuda_only_login_node_test():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    frozen = set(plan["login_cpu_cuda_deselects"])
    discovered = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if "reason=\"CUDA is unavailable\"" not in line:
                continue
            for following in lines[index + 1 : index + 6]:
                stripped = following.strip()
                if stripped.startswith("def test_"):
                    name = stripped.split("(", 1)[0].removeprefix("def ")
                    discovered.add(f"tests/{path.name}::{name}")
                    break
    assert discovered == frozen


def test_future_archive_lists_p95_after_p94_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    p94 = '"phase_9_p94_gradient_consistency.json"'
    p95_md = '"phase_9_p95_h100_qualification_plan.md"'
    p95_json = '"phase_9_p95_h100_qualification_plan.json"'

    assert source.count(p95_md) == 1
    assert source.count(p95_json) == 1
    assert source.index(p94) < source.index(p95_md) < source.index(p95_json)
    assert json.loads(PLAN.read_text(encoding="utf-8"))["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
