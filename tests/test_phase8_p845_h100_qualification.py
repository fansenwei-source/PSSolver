"""P8.4.5 H100-closure helper, analyzer, and frozen-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_plane_static_lifting_qualification import analyze
from benchmarks.qualify_plane_static_lifting import (
    LiftingProfileConfig,
    run_manufactured,
    run_profile,
    run_restart,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_h100_qualification_plan.json"
)


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_p845_plan_freezes_narrow_scope_and_keeps_neumann_deferred():
    plan = _plan()
    assert plan["phase"] == "P8.4.5"
    assert plan["classification"] == (
        "READY_P8_4_5_H100_QUALIFICATION_NOT_EXECUTED"
    )
    assert plan["authorization"] == {
        "compiled_runtime_promotion": False,
        "nonhomogeneous_neumann": False,
        "p8_4_5_h100_submission": True,
        "p8_5": False,
        "phase_9": False,
        "production_default_change": False,
    }
    contract = plan["qualification_contract"]
    assert contract["profile_report_count"] == 12
    assert contract["profile_shapes"] == [[128, 128, 32], [320, 320, 80]]
    assert contract["forward_transforms_per_step"] == 7.0
    assert contract["inverse_transforms_per_step"] == 32.0


@pytest.mark.parametrize(
    ("variant", "has_lifting"),
    (("homogeneous_control", False), ("strong_planar_lifting", True)),
)
def test_profile_helper_uses_real_runtime_and_counts_only_timesteps(
    variant,
    has_lifting,
):
    report = run_profile(
        LiftingProfileConfig(
            variant=variant,
            shape=(8, 8, 6),
            pointwise_execution="eager",
            warmup_steps=1,
            profile_steps=2,
        )
    )
    assert report["finite"] is True
    assert report["completed_steps"] == 3
    assert report["runtime_identity"]["effective"] == "legacy_production"
    assert report["transform_calls"]["forward_per_step"] == 7.0
    assert report["transform_calls"]["inverse_per_step"] == 32.0
    assert (report["lifting"] is not None) is has_lifting
    assert (report["wall_residual"] is not None) is has_lifting
    if has_lifting:
        assert report["wall_residual"]["max_linf"] <= 2e-16


def test_manufactured_helper_recovers_second_order_rate():
    report = run_manufactured(device="cpu")
    assert report["finite"] is True
    assert report["minimum_rate"] > 1.99


def test_restart_helper_is_byte_exact_for_physical_and_evolved_state():
    report = run_restart(
        shape=(8, 8, 6),
        device="cpu",
        segment_steps=1,
        final_steps=2,
        seed=24,
        pointwise_execution="eager",
    )
    assert report["finite"] is True
    assert report["all_byte_identical"] is True
    assert report["continuous_physical_sha256"] == (
        report["resumed_physical_sha256"]
    )
    assert report["lifting_restart"]["representation"] == (
        "homogeneous_remainder"
    )


def _synthetic_profile(
    *,
    shape: tuple[int, int, int],
    trial: int,
    variant: str,
    timestep: float,
) -> dict[str, object]:
    lifting = variant == "strong_planar_lifting"
    return {
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_profile",
        "config": {"shape": list(shape), "trial": trial, "variant": variant},
        "runtime_identity": {
            "requested": "legacy_production",
            "effective": "legacy_production",
            "fallback_used": False,
        },
        "finite": True,
        "git": {"head": "candidate"},
        "pssolver_import": "/installed/pssolver/__init__.py",
        "transform_calls": {"forward_per_step": 7.0, "inverse_per_step": 32.0},
        "pointwise_compile": {
            "execution": {
                "requested": "compile",
                "effective": "compile",
                "fallback_allowed": False,
            },
            "during_profile": {},
        },
        "lifting": {"representation": "homogeneous_remainder"} if lifting else None,
        "wall_residual": {"max_linf": 1e-16} if lifting else None,
        "timing": {
            "mean_timestep_seconds": timestep,
            "median_timestep_seconds": timestep,
        },
        "memory": {
            "peak_allocated_bytes": 105 if lifting else 100,
            "peak_reserved_bytes": 105 if lifting else 100,
        },
        "initial_q_sha256": f"initial-{shape}-{trial}",
    }


def _synthetic_reports() -> list[dict[str, object]]:
    reports = []
    for shape in ((128, 128, 32), (320, 320, 80)):
        for trial in range(1, 4):
            reports.append(
                _synthetic_profile(
                    shape=shape,
                    trial=trial,
                    variant="homogeneous_control",
                    timestep=1.0,
                )
            )
            reports.append(
                _synthetic_profile(
                    shape=shape,
                    trial=trial,
                    variant="strong_planar_lifting",
                    timestep=1.05,
                )
            )
    reports.extend(
        (
            {
                "phase": "P8.4.5",
                "kind": "plane_static_lifting_manufactured_convergence",
                "device": "cuda",
                "finite": True,
                "minimum_rate": 1.99,
            },
            {
                "phase": "P8.4.5",
                "kind": "plane_static_lifting_restart",
                "device": "cuda",
                "pointwise_execution": "compile",
                "finite": True,
                "all_byte_identical": True,
                "continuous_physical_sha256": "same",
                "resumed_physical_sha256": "same",
                "lifting_restart": {"representation": "homogeneous_remainder"},
            },
        )
    )
    return reports


def test_analyzer_accepts_complete_frozen_evidence():
    result = analyze(
        _plan(),
        _synthetic_reports(),
        expected_commit="candidate",
        expected_package_root=Path("/installed"),
    )
    assert result["classification"] == (
        "PASS_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE"
    )
    assert result["qualification_complete"] is True
    assert result["nonhomogeneous_neumann_supported"] is False


def test_analyzer_fails_closed_on_transform_or_restart_regression():
    reports = _synthetic_reports()
    reports[0]["transform_calls"]["forward_per_step"] = 8.0
    with pytest.raises(RuntimeError, match="forward transform"):
        analyze(
            _plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )

    reports = _synthetic_reports()
    reports[-1]["all_byte_identical"] = False
    with pytest.raises(RuntimeError, match="restart is not exact"):
        analyze(
            _plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )
