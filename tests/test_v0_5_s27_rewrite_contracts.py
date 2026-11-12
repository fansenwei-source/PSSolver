"""R-0093 G1–G6: retain the v0.4 normalization and restore contracts."""

from types import SimpleNamespace

import pytest

from pssolver.functional.legacy_declarations.legacy_restore import (
    _restore_plane_progress,
    restore_plane_state,
)


@pytest.mark.parametrize("interval,steps,counters", [(None, 23, (23, 0)), (20, 23, (3, 1))])
def test_g4_restore_progress_retains_v04_validation(interval, steps, counters):
    state = SimpleNamespace(completed_steps=0, refresh_phase=(0, 0))
    _restore_plane_progress(state, interval, steps, interval, *counters)
    assert (state.completed_steps, state.refresh_phase) == (steps, counters)
    for stored_interval, stored_counts in [(7, counters), (interval, (9, 9))]:
        with pytest.raises(RuntimeError, match="^restored spectral-refresh counters are inconsistent$"):
            _restore_plane_progress(state, interval, steps, stored_interval, *stored_counts)
        assert (state.completed_steps, state.refresh_phase) == (steps, counters)


def test_g1_rejects_wrong_engine_before_reading_any_checkpoint():
    with pytest.raises(TypeError, match="^engine must be an Engine$"):
        restore_plane_state(object(), object(), object(), run_spec=object())
