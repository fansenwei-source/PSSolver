from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.analyze_rc4_plane_nyquist_qualification import (
    FROZEN_INPUT_PATHS,
    GRIDS,
    MEMORY_RATIO_LIMIT,
    MEAN_RATIO_LIMIT,
    MEDIAN_RATIO_LIMIT,
    PAIRED_RATIO_LIMIT,
    TRIALS,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "notes" / "PSSolver_v0_2_0rc4_rc412_h100_qualification_plan.json"


def _plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc412_plan_freezes_runtime_commits_and_scope():
    plan = _plan()

    assert plan["runtime_identity"]["baseline"]["commit"] == (
        "0ce1e898d0b46f74e1032a3a7e93b46ea8667692"
    )
    assert plan["runtime_identity"]["candidate"]["commit"] == (
        "d358ed66fe75da68a3e17fed402a956f21f1dd37"
    )
    assert plan["runtime_identity"]["qualification_support_must_not_change_runtime_sources"] is True
    assert plan["h100_execution"]["production_default_changed"] is False
    assert "PSSolver-Control modification" in plan["not_authorized"]
    assert "nematics3d modification" in plan["not_authorized"]


def test_rc412_plan_binds_versioned_support_files():
    support = _plan()["qualification_support"]

    for name in ("profile_runner", "cuda_diagnostic", "analyzer", "cuda_test"):
        path = ROOT / support[name]
        assert path.is_file()
        assert _sha256(path) == support[f"{name}_sha256"]


def test_rc412_plan_freezes_inputs_and_profile_matrix():
    plan = _plan()

    assert set(plan["frozen_inputs"]) == set(GRIDS)
    for grid, shape in GRIDS.items():
        frozen = plan["frozen_inputs"][grid]
        assert frozen["path"] == FROZEN_INPUT_PATHS[grid]
        assert frozen["shape"] == [*shape, 5]
        assert frozen["dtype"] == "float64"
        assert len(frozen["sha256"]) == 64

    matrix = plan["profile_matrix"]
    assert matrix["profile_count"] == 2 * len(GRIDS) * len(TRIALS)
    assert matrix["trials_per_role_and_grid"] == len(TRIALS)
    assert matrix["balanced_order"] == ["A/B", "B/A", "A/B"]
    assert matrix["forward_transforms_per_step"] == 7.0
    assert matrix["inverse_transforms_per_step"] == 32.0
    assert matrix["cross_role_final_sha_identity_required"] is False


def test_rc412_plan_matches_analyzer_thresholds():
    thresholds = _plan()["thresholds"]

    assert thresholds["mean_candidate_over_baseline_timestep_max"] == MEAN_RATIO_LIMIT
    assert thresholds["median_candidate_over_baseline_timestep_max"] == MEDIAN_RATIO_LIMIT
    assert thresholds["each_paired_candidate_over_baseline_timestep_max"] == PAIRED_RATIO_LIMIT
    assert thresholds["peak_allocated_candidate_over_baseline_max"] == MEMORY_RATIO_LIMIT
    assert thresholds["peak_reserved_candidate_over_baseline_max"] == MEMORY_RATIO_LIMIT


def test_rc412_plan_requires_exact_cuda_nodes_and_single_job():
    plan = _plan()
    nodes = plan["cuda_only_node_ids"]

    assert len(nodes) == 7
    assert len(set(nodes)) == 7
    assert plan["cpu_gate"]["cuda_only_deselect_count"] == len(nodes)
    assert nodes[-1] == (
        "tests/test_rc4_plane_nyquist_cuda.py::"
        "test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence"
    )
    execution = plan["h100_execution"]
    assert execution["formal_submission_limit"] == 1
    assert execution["automatic_retry"] is False
    assert execution["second_job_without_new_authorization"] is False
    assert execution["tf32"] is False
    assert plan["status"] == "LOCAL_CPU_GATE_PASS_READY_FOR_SINGLE_H100"
    assert plan["cpu_gate"]["result"] == {
        "passed": 2781,
        "deselected_cuda_only": 7,
        "subtests_passed": 8,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "elapsed_seconds": 199.90,
    }


def test_rc412_plan_lists_every_current_cuda_only_login_node_test():
    frozen = set(_plan()["cuda_only_node_ids"])
    discovered = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
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
