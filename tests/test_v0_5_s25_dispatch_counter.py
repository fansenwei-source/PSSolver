"""Exact rules for the R-0078 dispatcher-level K2 measurements."""

import subprocess
import sys
from functools import wraps
from pathlib import Path

import torch

from qualification.k2.dispatch_counter import DispatchCounter
from qualification.k2.resident_storage import ResidentStorageCounter


def isolated(function):
    """Run every measurement test in a fresh CPU process, never a compile survivor."""

    @wraps(function)
    def run():
        root = Path(__file__).resolve().parents[1]
        code = (
            f"import sys; sys.path.insert(0, {str(root)!r}); "
            f"from tests import test_v0_5_s25_dispatch_counter as cases; "
            f"cases.{function.__name__}.__wrapped__()"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


@isolated
def test_views_inplace_and_out_are_not_allocations():
    tensor = torch.arange(16, dtype=torch.float64)
    output = torch.empty_like(tensor)
    counter = DispatchCounter()
    with counter:
        tensor.view(4, 4)
        tensor[::2]
        tensor.add_(1)
        torch.add(tensor, tensor, out=output)
    assert counter.counts() == dict(transforms=0, allocations=0, scalar_extractions=0)


@isolated
def test_all_tensor_outputs_are_counted_and_empty_storage_is_excluded():
    tensor = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    counter = DispatchCounter()
    with counter:
        tensor.clone()
        torch.max(tensor, dim=0)
        torch.empty(0)
    assert counter.allocations == 3


@isolated
def test_transform_and_scalar_counts():
    tensor = torch.arange(16, dtype=torch.float64)
    counter = DispatchCounter()
    with counter:
        spectrum = torch.fft.rfft(tensor)
        torch.fft.irfft(spectrum)
        tensor[0].item()
    assert counter.transforms == 2
    assert counter.scalar_extractions == 1


@isolated
def test_counter_does_not_keep_storage_alive():
    counter = ResidentStorageCounter()
    retained = torch.empty(17, dtype=torch.float64)
    released = torch.empty(19, dtype=torch.float64)
    del released
    assert counter.snapshot()["resident_bytes"] == 17 * 8
    del retained
    assert counter.snapshot()["resident_bytes"] == 0


@isolated
def test_initial_storages_are_excluded_and_strongly_held():
    initial = torch.empty(23, dtype=torch.float64)
    storage = initial.untyped_storage()
    identity = storage._cdata
    counter = ResidentStorageCounter()
    assert counter._initial[identity] is storage
    del initial, storage
    assert counter._initial[identity].nbytes() == 23 * 8
    assert counter.snapshot()["resident_bytes"] == 0


@isolated
def test_views_share_one_live_storage_and_record_all_owners():
    counter = ResidentStorageCounter()
    tensor = torch.empty(15, dtype=torch.float32)
    view = tensor.view(3, 5)
    snapshot = counter.snapshot()
    assert snapshot["resident_bytes"] == 15 * 4
    rows = [row for row in snapshot["live_storages"] if not row["initial"]]
    assert len(rows) == 1
    assert rows[0]["_cdata"] == view.untyped_storage()._cdata
    assert {tuple(owner["shape"]) for owner in rows[0]["owners"]} == {(15,), (3, 5)}


@isolated
def test_reused_address_has_current_size_not_released_allocation_size():
    backing = bytearray(64)
    counter = ResidentStorageCounter()
    released = torch.frombuffer(backing, dtype=torch.uint8, count=1)
    address = released.untyped_storage().data_ptr()
    assert released.untyped_storage().nbytes() == 1
    del released
    replacement = torch.frombuffer(backing, dtype=torch.float64)
    assert replacement.untyped_storage().data_ptr() == address
    snapshot = counter.snapshot()
    assert snapshot["resident_bytes"] == 64
    assert [row["nbytes"] for row in snapshot["live_storages"] if not row["initial"]] == [64]


@isolated
def test_distinct_storage_identities_at_one_address_are_not_conflated():
    backing = bytearray(64)
    initial = torch.frombuffer(backing, dtype=torch.uint8, count=1)
    counter = ResidentStorageCounter()
    replacement = torch.frombuffer(backing, dtype=torch.float64)
    assert replacement.untyped_storage().data_ptr() == initial.untyped_storage().data_ptr()
    assert replacement.untyped_storage()._cdata != initial.untyped_storage()._cdata
    assert counter.snapshot()["resident_bytes"] == 64
