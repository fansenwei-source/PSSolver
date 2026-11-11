"""Byte-for-byte CPU references; mismatched prerequisite is a failure, not skip."""

import json
from pathlib import Path

import pytest

from qualification.identity.analyze import compact, digest
from qualification.identity.channel_comparison import (
    ChannelEngineCase,
    channel_tensor_progress,
    comparable_channel,
    trace_channel,
)
from qualification.identity.checkpoint_fixtures import continue_case
from qualification.identity.historical_continuations import replay
from qualification.identity.periodic_comparison import compact_periodic
from qualification.identity.plane_comparison import compact_plane
from qualification.identity.records import fingerprint, require_fingerprint
from qualification.identity.runtime_case_v2 import RuntimeCaseV2 as RuntimeCase
from qualification.identity.sampling import trace
from qualification.identity.vjp import vjp_record


ROOT = Path(__file__).parent
BASELINES = ROOT / "baselines/v0_4"
TABLE = json.loads((BASELINES / "trajectories.json").read_text())
FIXTURES = ROOT / "fixtures/v0_4_0_s23"
CHECKPOINT = json.loads((BASELINES / "checkpoint_continuations.json").read_text())
PERIODIC_V2 = json.loads((ROOT / "baselines/s25/periodic_comparison.json").read_text())
PLANE_V2 = json.loads((ROOT / "baselines/s27/plane_comparison.json").read_text())
FIXTURE_MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
HISTORICAL = json.loads((BASELINES / "historical_continuations.json").read_text())
CHANNEL_V2 = json.loads(
    (ROOT.parent / "qualification/s28a_channel_release_comparison.json").read_text()
)


@pytest.fixture(autouse=True)
def baseline_prerequisite():
    require_fingerprint(fingerprint(), TABLE["fingerprint"])


@pytest.mark.parametrize("name", sorted(TABLE["cases"]))
def test_replay_recorded_release_trajectory(name, tmp_path):
    expected = TABLE["cases"][name]
    scene = expected["scene"]
    if scene["family"] == "channel":
        _check_channel_trajectory(name, scene, expected, tmp_path)
        return
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
        result.update(
            disposition="accepted", trace=trace(scene, tmp_path, case_factory=RuntimeCase)
        )
        if scene["family"].startswith("functional_"):
            result["vjp"] = vjp_record(scene, tmp_path)
    if scene["family"] == "periodic":
        if result["disposition"] == "accepted":
            assert all(
                not contract["raw_metadata_differences"]
                for contract in result["trace"]["restart_contracts"].values()
            )
        assert compact_periodic(result) == PERIODIC_V2["cases"][name]
    elif scene["family"] == "plane":
        assert digest(compact_plane(result)) == PLANE_V2["cases"][name]
    else:
        assert compact(result) == expected


def _check_channel_trajectory(name, scene, expected, tmp_path):
    if expected["disposition"] == "rejected":
        with pytest.raises((ValueError, TypeError, NotImplementedError)) as error:
            ChannelEngineCase(scene, tmp_path)
        assert f"{type(error.value).__module__}.{type(error.value).__name__}" == expected[
            "exception_type"
        ]
        assert str(error.value) == expected["message"]
        return
    actual = comparable_channel(trace_channel(scene, tmp_path))
    # The stored regression digest has no stable checkout path. V2 compares
    # the entire identity separately at an identical absolute work path.
    del actual["identity"]["configuration"]
    assert digest(actual) == CHANNEL_V2["trajectories"][name]["trace_sha256"]


@pytest.mark.parametrize("scene", FIXTURE_MANIFEST["scenes"], ids=lambda item: item["id"])
def test_released_checkpoint_sixteen_step_continuation(scene, tmp_path):
    if scene["family"] == "channel":
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        case = ChannelEngineCase(scene, runtime)
        case.restore(FIXTURES / scene["id"], 2, legacy=True)
        assert case.state.solver["pressure_guess_action"] is None
        assert case.state.solver["pressure_guess_action_exponent"] is None
        rows = [case.record()]
        for _ in range(16):
            case.step()
            rows.append(case.record())
        assert [digest(channel_tensor_progress(row)) for row in rows] == CHANNEL_V2[
            "legacy_continuations"
        ][scene["id"]]
        return
    actual = continue_case(
        scene, FIXTURES / scene["id"], tmp_path / "runtime", case_factory=RuntimeCase
    )
    assert [digest(row) for row in actual] == CHECKPOINT["cases"][scene["id"]]


def test_retained_historical_families_continue_sixteen_steps(tmp_path):
    actual = replay(ROOT / "fixtures/v0_2_0_checkpoints_v1", tmp_path)
    assert {
        key: [digest(row) for row in rows] for key, rows in actual["cases"].items()
    } == HISTORICAL["cases"]
