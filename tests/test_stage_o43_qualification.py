"""CPU tests for the bounded Stage O.4.3 qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from pssolver.experimental.stage_o43_plan import build_stage_o43_h100_plan
from pssolver.experimental.stage_o43_qualification import (
    analyze_stage_o43_qualification,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> Path:
    return _write_json(
        path,
        {
            "qualification_stage": "O.4.2",
            "classification": "DIAGNOSTIC_COMPLETE",
            "accounting_ready_for_optimization": True,
            "eligible_for_stage_o43_optimization_design": True,
            "production_default_changed": False,
        },
    )


def _trajectory(
    root: Path,
    *,
    role: str,
    perturbation: float = 0.0,
) -> Path:
    root.mkdir()
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    metadata = {
        "qualification_stage": "O.4.3",
        "classification": "TRAJECTORY_COMPLETE",
        "measurement_role": role,
        "completed_steps": 100,
        "saved_steps": [0, 100],
        "production_default_changed": False,
        "production_metadata_sha256": "1" * 64,
        "production_initial_q_sha256": "2" * 64,
    }
    _write_json(root / "stage_o43_metrics.json", metadata)
    shapes = {"Q": (2, 2, 2, 5), "u": (2, 2, 2, 3), "p": (2, 2, 2)}
    for step in (0, 100):
        for field, shape in shapes.items():
            values = np.ones(shape, dtype=np.float64)
            if role == "candidate" and step == 100:
                values.flat[0] += perturbation
            with (root / f"{field}_{step}.npy").open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
    return root


def _profile(
    path: Path,
    *,
    role: str,
    shape: tuple[int, int, int],
    input_digit: str,
    timestep: float,
    peak: int,
) -> Path:
    candidate = role == "candidate"
    assembly = {
        "policy": {
            "mode": (
                "contiguous_storage_view" if candidate else "copy_cat"
            )
        },
        "contiguous_view_batches": 5 if candidate else 0,
        "copy_cat_batches": 5 if candidate else 10,
        "retained_tensor_references": 0,
    }
    return _write_json(
        path,
        {
            "qualification_stage": "O.4.3",
            "classification": "PROFILE_COMPLETE",
            "measurement_role": role,
            "production_default_changed": False,
            "production_metadata_sha256": input_digit * 64,
            "production_initial_q_sha256": "2" * 64,
            "configuration": {
                "shape": list(shape),
                "warmup_steps": 10,
                "profile_steps": 20,
            },
            "environment": {
                "device_name": "NVIDIA H100 PCIe",
                "cuda_matmul_allow_tf32": False,
            },
            "algebraic_output_publication_policy": {
                "mode": (
                    "preallocated_packed" if candidate else "deferred_stack"
                )
            },
            "projected_batch_assembly_policy": {
                "mode": (
                    "contiguous_storage_view" if candidate else "copy_cat"
                )
            },
            "batch_assembly_diagnostics": assembly,
            "throughput": {"mean_timestep_seconds": timestep},
            "memory": {"peak_allocated_bytes": peak},
            "finite": True,
        },
    )


def _qualification_inputs(tmp_path: Path, *, perturbation: float = 1.0e-13):
    evidence = _evidence(tmp_path / "o42.json")
    trajectories = {
        role: _trajectory(
            tmp_path / f"trajectory_{role}",
            role=role,
            perturbation=perturbation,
        )
        for role in ("baseline", "candidate")
    }
    profiles = {}
    for scale, shape, digit in (
        ("r128", (128, 128, 32), "1"),
        ("r320", (320, 320, 80), "3"),
    ):
        for role in ("baseline", "candidate"):
            profiles[(scale, role)] = [
                _profile(
                    tmp_path / f"{scale}_{role}_{trial}.json",
                    role=role,
                    shape=shape,
                    input_digit=digit,
                    timestep=0.1 if role == "baseline" else 0.09,
                    peak=1000 if role == "baseline" else 900,
                )
                for trial in range(3)
            ]
    return evidence, trajectories, profiles


def test_stage_o43_analysis_recommends_only_a_safe_measured_improvement(tmp_path):
    evidence, trajectories, profiles = _qualification_inputs(tmp_path)
    report = analyze_stage_o43_qualification(
        trajectories["baseline"],
        trajectories["candidate"],
        profiles[("r128", "baseline")],
        profiles[("r128", "candidate")],
        profiles[("r320", "baseline")],
        profiles[("r320", "candidate")],
        stage_o42_report=evidence,
        expected_stage_o42_sha256=_sha256(evidence),
    )

    assert report["classification"] == "A_recommended"
    assert report["eligible_for_stage_o44_decision"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False
    assert all(report["gates"].values())
    assert report["trajectory"]["q0_byte_identical"] is True


def test_stage_o43_analysis_rejects_a_scientifically_different_candidate(tmp_path):
    evidence, trajectories, profiles = _qualification_inputs(
        tmp_path,
        perturbation=1.0,
    )
    report = analyze_stage_o43_qualification(
        trajectories["baseline"],
        trajectories["candidate"],
        profiles[("r128", "baseline")],
        profiles[("r128", "candidate")],
        profiles[("r320", "baseline")],
        profiles[("r320", "candidate")],
        stage_o42_report=evidence,
        expected_stage_o42_sha256=_sha256(evidence),
    )

    assert report["classification"] == "C_rejected"
    assert report["gates"]["numerical_equivalence"] is False
    assert report["eligible_for_stage_o44_decision"] is False


def test_stage_o43_plan_is_bounded_balanced_and_non_promoting(tmp_path):
    reference_r128 = tmp_path / "r128"
    reference_r320 = tmp_path / "r320"
    reference_r128.mkdir()
    reference_r320.mkdir()
    evidence = _evidence(tmp_path / "o42.json")
    plan = build_stage_o43_h100_plan(
        project_root=PROJECT_ROOT,
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        python=sys.executable,
        r128_production_reference_dir=reference_r128,
        r320_production_reference_dir=reference_r320,
        expected_commit="a" * 40,
        stage_o42_report=evidence,
        expected_stage_o42_sha256=_sha256(evidence),
    )

    assert plan["planning_only"] is True
    assert len(plan["commands"]["trajectories"]) == 2
    assert len(plan["commands"]["profiles_balanced"]) == 12
    assert plan["candidate_contract"]["fallback"] == "copy_cat"
    assert plan["candidate_contract"]["changes_equations"] is False
    assert plan["candidate_contract"]["production_default_may_change"] is False
    assert plan["qualification_contract"]["trajectory_steps"] == 100
    assert plan["qualification_contract"]["eligible_result"] == (
        "stage_o44_decision_only"
    )
