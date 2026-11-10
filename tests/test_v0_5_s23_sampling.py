"""Sampling is explicit; changing events must not silently alter the reference."""

from copy import deepcopy

import pytest

from qualification.identity.analyze import validate_sampling
from qualification.identity.runtime_case_v2 import RuntimeCaseV2 as RuntimeCase
from qualification.identity.sampling import restart_contract, sample, trace
from qualification.identity.scenarios import scenarios
from tests._dynamic_flags import set_dynamic_flag


def scene(family):
    item = next(
        row
        for row in scenarios()
        if row["family"] == family and row["dtype"] == "float64" and not row["options"]
    )
    return {**item, "shape": [6, 6, 4], "steps": 5, "restart_steps": [2, 4]}


@pytest.mark.parametrize(
    "family", ["periodic", "plane", "channel", "functional_periodic", "functional_channel"]
)
def test_identical_event_schedule_restores_exact_fields_and_clocks(tmp_path, family):
    row = scene(family)
    recorded = trace(row, tmp_path, case_factory=RuntimeCase)
    validate_sampling({"disposition": "accepted", "scene": row, "trace": recorded})
    for start, replay in recorded["restarts"].items():
        if family == "periodic":
            assert recorded["restart_contracts"][start]["raw_metadata_differences"] == []
        changed = deepcopy(replay)
        changed[0]["after_checkpoint"]["step"] += 1
        assert not restart_contract(recorded["steps"], changed, int(start))[
            "field_bytes_and_clocks_equal"
        ]


def test_resumed_schedule_replays_later_checkpoint_events(tmp_path):
    class EventCase:
        scene = {"observe": True, "restart_steps": [2, 4], "steps": 5}
        events = []

        def record(self):
            return list(self.events)

        def observe(self):
            self.events.append("observe")

        def export(self, path):
            self.events.append(path.name)

        def step(self):
            self.events.append("step")

    case = EventCase()
    result = sample(case, tmp_path, 4, restart_from=2)
    assert result["after_checkpoint"] == ["observe", "replay_2_4"]
    assert result["after_step"][-1] == "step"


def test_pre_update_callback_records_the_rhs_input_after_synchronization(tmp_path, monkeypatch):
    case = RuntimeCase(scene("periodic"), tmp_path)
    calls = []
    update = case.prepare
    record = case.record

    def update_then_mark():
        update()
        calls.append("algebraic")

    def record_then_mark():
        calls.append("callback")
        return record()

    monkeypatch.setattr(case, "prepare", update_then_mark)
    monkeypatch.setattr(case, "record", record_then_mark)
    before = case.step()
    assert calls == ["algebraic", "callback"]
    # Engine's explicit prepare publishes the current constraint token (section 9).
    assert before["step"] == 0 and before["progress"]["constraints_current"]
    assert case.record()["step"] == 1


@pytest.mark.parametrize("original", [False, True])
def test_virtual_backend_flag_restores_getter_value_without_deletion(original):
    class VirtualFlag:
        def __init__(self):
            self.values = {"flag": original}

        def __getattr__(self, name):
            return self.values[name]

        def __setattr__(self, name, value):
            if name == "flag":
                self.values[name] = value
            else:
                object.__setattr__(self, name, value)

    class Finalizers:
        def __init__(self):
            self.callbacks = []

        def addfinalizer(self, callback):
            self.callbacks.append(callback)

    target, request = VirtualFlag(), Finalizers()
    set_dynamic_flag(request, target, "flag", not original)
    assert target.flag is not original
    assert "flag" not in target.__dict__
    for callback in reversed(request.callbacks):
        callback()
    assert target.flag is original
    assert "flag" not in target.__dict__
