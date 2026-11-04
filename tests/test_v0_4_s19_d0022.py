"""Exact D-0022 exceptions: raw bytes, not NaN-aware equality or tolerances."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks._s19_compare import (
    HardQualificationError, KNOWN_NONFINITE_PCG_RECORDS, compare_field,
    final_verdict, require_execution,
)
from benchmarks.plan_v0_4_s19_h100 import build_plan
from benchmarks.run_v0_4_s19_self_compare import _persist_tensor, _read_array, _compare

EXPECTED = {('fixed_float32_roundoff_floor', 'oracle', 'gradient'),
            ('fixed_float32_floor_crossing', 'oracle', 'gradient')}


def test_exception_inventory_is_exact_and_plan_frozen_to_the_same_two_records():
    assert KNOWN_NONFINITE_PCG_RECORDS == EXPECTED
    assert build_plan()['numerical']['known_nonfinite_records'] == [list(x) for x in sorted(EXPECTED)]


@pytest.mark.parametrize('key', sorted(EXPECTED))
def test_registered_nonfinite_roundtrip_and_all_three_pairings_are_raw_bytes(tmp_path, key):
    array = np.array([0x7fc00001, 0xff800000, 0x80000000, 0x3f800000], dtype=np.uint32).view(np.float32)
    reports = []
    for role in ('a', 'b', 'candidate'):
        saved = _persist_tensor(tmp_path / f'{role}.npy', torch.from_numpy(array.copy()), pcg_record=key)
        assert _read_array(saved).tobytes() == array.tobytes()
        reports.append({'kind': 'pcg', 'cases': [{'case': {'name': key[0]}, 'path': key[1],
                        'execution_device': 'cpu', 'fields': {'gradient': saved}, 'record': {}}]})
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert _compare(reports[a], reports[b])['classification'] == 'PASS'
    changed = array.copy()
    changed.view(np.uint32)[0] ^= 1  # same NaN location, different payload
    result = compare_field(array, changed, field='gradient', step=0, pcg_record=key)
    assert result['classification'] == 'N' and result['bitwise_equal'] is False
    assert result['nonfinite_positions_equal'] is True
    assert final_verdict([result, {'classification': 'PASS'}], formal_h100=True)['classification'] == 'FAIL_N'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('key', [
    ('fixed_float32_roundoff_floor', 'oracle', 'result'),
    ('fixed_float32_floor_crossing', 'oracle', 'result'),
    ('fixed_float32_roundoff_floor', 'production', 'gradient'),
    ('fixed_float32_floor_crossing', 'transpose', 'gradient'),
    ('fixed_float32_roundoff_floor', 'implicit', 'gradient'),
    ('fixed_float64', 'oracle', 'gradient'),
    ('fixed_float32_roundoff_floor_extra', 'oracle', 'gradient'),
    ('fixed_float32_floor_crossing', 'oracle', 'warm_pressure'), None,
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_injection_anywhere_else_stops_before_publication(tmp_path, key, value):
    output = tmp_path / 'rejected.npy'
    with pytest.raises(HardQualificationError, match='non-finite'):
        _persist_tensor(output, torch.tensor([value]), pcg_record=key)
    assert not output.exists()
    with pytest.raises(HardQualificationError, match='non-finite'):
        compare_field(np.array([1.]), np.array([value]), field='injected', step=0, pcg_record=key)


def test_execution_does_not_lie_about_finiteness_or_accept_expanded_exceptions():
    report = dict(kind='pcg', finite=False, installed_wheel_only=True,
                  cache_initially_empty=True, fallback_used=False, returncode=0,
                  nonfinite_records=[list(x) for x in sorted(EXPECTED)])
    require_execution(report, {'kind': 'pcg'})
    report['nonfinite_records'].append(['fixed_float64', 'oracle', 'gradient'])
    with pytest.raises(HardQualificationError):
        require_execution(report, {'kind': 'pcg'})


def test_v63_is_recorded_separately_from_the_s19_support_exception():
    # Freeze the S-19 snapshot; the S-20 decision may advance the living entry.
    record = json.loads((Path(__file__).parents[1] / 'notes/PSSolver_v0_3_0_s03_successor_128_r0064_import.json').read_text())
    issue = record['registry_snapshot']['V-63']
    assert issue['severity'] == 'low' and issue['target_release'] == '0.4 S-20'
    assert issue['status'] in {'open', 'claimed_fixed', 'verified', 'accepted'}
    assert issue['affects_control_paths'] is False
