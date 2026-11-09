"""Current diagnostics contracts split unchanged from the O.4.1 audit."""

import pytest

from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window


def test_cuda_memory_helpers_have_a_complete_cpu_schema():
    snapshot = cuda_memory_snapshot("cpu")
    assert snapshot == {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    assert all(value is None for value in cuda_memory_window(snapshot, snapshot, snapshot).values())


def test_cuda_memory_window_rejects_an_impossible_peak():
    start = {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
    }
    end = {"allocated_bytes": 12, "reserved_bytes": 20}
    peak = {"peak_allocated_bytes": 11, "peak_reserved_bytes": 20}
    with pytest.raises(ValueError, match="peak allocated"):
        cuda_memory_window(start, end, peak)
