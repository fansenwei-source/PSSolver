"""D-0069 changes the comparison source for exactly three cache fields."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from qualification.identity.records import fingerprint, require_fingerprint
from qualification.identity.sampling import restart_contract
from qualification.s28b_trajectory import compare_trajectory


SCENE = "channel_default_float64_relative_tolerance"


def test_s23_comparisons_require_the_frozen_cpu_environment():
    path = Path(__file__).with_name("baselines") / "v0_4/trajectories.json"
    require_fingerprint(fingerprint(), json.loads(path.read_bytes())["fingerprint"])


def _row(step, action):
    return {
        "step": step,
        "tensors": {"Qxx": {"sha256": "unchanged"}},
        "progress": {"step_count": step},
        "solver": {
            "operator_action": {"sha256": action},
            "operator_action_matches_guess": True,
            "operator_action_working_exponent": 0,
            "pressure_iterations": 5,
        },
    }


def _records(resumed_action, *, iterations=5):
    continuous = [
        {"after_checkpoint": _row(0, "initial")},
        {"before_observation": _row(1, "pre-checkpoint"), "after_checkpoint": _row(1, "saved")},
    ]
    restored = _row(1, resumed_action)
    restored["solver"]["pressure_iterations"] = iterations
    resumed = [{"before_observation": deepcopy(restored), "after_checkpoint": restored}]
    return {
        "steps": continuous,
        "restarts": {"1": resumed},
        "restart_contracts": {"1": restart_contract(continuous, resumed, 1)},
    }


def test_first_resumed_cache_matches_written_checkpoint_not_the_pre_observation_cache():
    result = compare_trajectory(SCENE, _records("saved"), _records("legacy-cache"))
    assert result["differences"] == []


def test_a_bad_resume_cache_fails_even_when_the_sealed_cache_has_the_same_value():
    with pytest.raises(ValueError, match="D-0069 cache mismatch"):
        compare_trajectory(SCENE, _records("bad"), _records("bad"))


def test_pressure_iterations_and_other_scenes_still_compare_to_the_sealed_record():
    original = _records("legacy-cache")
    result = compare_trajectory(SCENE, _records("saved", iterations=6), original)
    assert any(path.endswith("/pressure_iterations") for path in result["differences"])
    result = compare_trajectory("channel_default_float64_fixed_work", _records("saved"), original)
    assert any("/operator_action/" in path for path in result["differences"])
