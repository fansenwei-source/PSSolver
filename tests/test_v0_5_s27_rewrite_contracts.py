"""R-0093 G1–G6: retain the v0.4 normalization and restore contracts."""

from types import SimpleNamespace

import pytest
import torch

from pssolver.engine.groups import _normalize_field_indices, select_group, store_group
from pssolver.Field import Fields
from pssolver.runtime.legacy_entry import _restore_plane_progress, restore_plane_state


def _outcome(operation):
    try:
        return ("value", operation())
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        return (type(exc).__name__, str(exc))


@pytest.mark.parametrize("keys", [[0, 1, 2], [], [-1], [3], ["ux", "uy", "p"], [0.5], ["unknown"]])
def test_g6_normalization_keeps_original_result_or_error(keys):
    original = Fields.__new__(Fields)
    original.name_to_idx = {"ux": 0, "uy": 1, "p": 2}
    original.boundary_conditions = (None,) * 3
    assert _outcome(lambda: _normalize_field_indices(keys, original.name_to_idx, 3)) == (
        _outcome(lambda: original._normalize_field_indices(keys))
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_g6_advanced_group_preserves_v1_gather_and_scatter(dtype):
    original = Fields.__new__(Fields)
    original.name_to_idx = {"ux": 0, "uy": 1, "uz": 2, "p": 3}
    original.boundary_conditions = (None,) * 4
    original.transform_group_indexing = "contiguous_slice"
    original.spectral = torch.arange(32, dtype=dtype).reshape(4, 1, 2, 4)
    original.spatial = original.spectral.clone()
    actual = original.spatial.clone()
    keys = ["ux", "uy", "p"]
    selected = select_group(original.spectral, keys, original.name_to_idx, 4, "contiguous_slice")
    assert torch.equal(selected, original.select_spectral_group(keys))
    values = selected + 2
    store_group(actual, keys, values, original.name_to_idx, 4, "contiguous_slice")
    original.store_spatial_group(keys, values)
    assert torch.equal(actual, original.spatial)


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
