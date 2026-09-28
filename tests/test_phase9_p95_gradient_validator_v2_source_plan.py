"""Static contract tests for the formal P9.5 gradient validator v2."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver.functional.validation as validation
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN = NOTES / "phase_9_p95_gradient_validator_v2_source_plan.json"
ARCHIVE = NOTES / "build_verbatim_archive_pdf.py"


def test_p95_validator_v2_source_plan_matches_implementation_contract():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["phase"] == "P9.5-gradient-validator-v2-source"
    assert plan["status"] == "implemented_pending_h100_closure"
    assert plan["validator"] == {
        "format_version": 2,
        "objective": "fixed_quadratic_dynamics_plus_observations",
        "direction": "low_mode",
        "direction_is_independent_of_autograd": True,
        "finite_difference_method": "paired_quadratic",
        "state_epsilon": 1.0e-5,
        "activity_epsilon": 1.0e-2,
        "relative_tolerance": 2.0e-5,
        "direction_cosine_requirement": "finite_and_strictly_positive",
        "gradient_path_count": 6,
        "all_gradient_paths_must_be_finite_nonzero_and_pass": True,
        "activity_perturbation_must_remain_admissible": True,
    }
    assert validation.PERIODIC_GRADIENT_VALIDATION_VERSION == 2
    assert validation._frozen_gradient_contract(torch.float64) == (
        1.0e-5,
        1.0e-2,
        2.0e-5,
    )
    assert plan["required_closure"]["matrix_profile_count"] == 18
    assert plan["required_closure"]["batch_sizes_qualified_on_pass"] == [1]
    assert plan["required_closure"]["larger_batch_qualified_on_pass"] is False
    assert plan["stopping_rules"]["p9_5_complete_before_h100_closure"] is False
    assert plan["stopping_rules"]["eligible_for_p9_6_before_h100_closure"] is False


def test_p95_validator_v2_source_plan_is_in_verbatim_archive_manifest():
    source = ARCHIVE.read_text(encoding="utf-8")

    assert '"phase_9_p95_gradient_validator_v2_source_plan.md"' in source
    assert '"phase_9_p95_gradient_validator_v2_source_plan.json"' in source
