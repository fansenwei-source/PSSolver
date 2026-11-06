"""Byte-for-byte CPU references; mismatched prerequisite is a failure, not skip."""

import json
from pathlib import Path

import pytest

from qualification.identity.analyze import compact, digest
from qualification.identity.checkpoint_fixtures import continue_case
from qualification.identity.historical_continuations import replay
from qualification.identity.records import fingerprint, require_fingerprint
from qualification.identity.runtime_case import RuntimeCase
from qualification.identity.sampling import trace
from qualification.identity.vjp import vjp_record


ROOT = Path(__file__).parent
BASELINES = ROOT / "baselines/v0_4"
TABLE = json.loads((BASELINES / "trajectories.json").read_text())
FIXTURES = ROOT / "fixtures/v0_4_0_s23"
CHECKPOINT = json.loads((BASELINES / "checkpoint_continuations.json").read_text())
FIXTURE_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
HISTORICAL = json.loads((BASELINES / "historical_continuations.json").read_text())


@pytest.fixture(autouse=True)
def baseline_prerequisite():
    require_fingerprint(fingerprint(), TABLE["fingerprint"])


@pytest.mark.parametrize("name", sorted(TABLE["cases"]))
def test_replay_recorded_release_trajectory(name, tmp_path):
    expected = TABLE["cases"][name]
    scene = expected["scene"]
    result = {"scene": scene}
    try:
        RuntimeCase(scene, tmp_path)
    except (ValueError, TypeError, NotImplementedError) as error:
        result.update(
            disposition="rejected",
            exception_type=f"{type(error).__module__}.{type(error).__name__}",
            message=str(error),
        )
    else:
        result.update(disposition="accepted", trace=trace(scene, tmp_path))
        if scene["family"].startswith("functional_"):
            result["vjp"] = vjp_record(scene, tmp_path)
    assert compact(result) == expected


@pytest.mark.parametrize("scene", FIXTURE_MANIFEST["scenes"], ids=lambda item: item["id"])
def test_released_checkpoint_sixteen_step_continuation(scene, tmp_path):
    actual = continue_case(scene, FIXTURES / scene["id"], tmp_path / "runtime")
    assert [digest(row) for row in actual] == CHECKPOINT["cases"][scene["id"]]


def test_retained_historical_families_continue_sixteen_steps(tmp_path):
    actual = replay(ROOT / "fixtures/v0_2_0_checkpoints_v1", tmp_path)
    assert {
        key: [digest(row) for row in rows] for key, rows in actual["cases"].items()
    } == HISTORICAL["cases"]
