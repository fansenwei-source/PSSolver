"""Failure contracts remain identical after a failed fixed-work floor retry."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pssolver.linear_solvers.stokes._channel_pcg_control import ChannelPCGControl


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('acceptable', [False, True])
@pytest.mark.parametrize('represented', [False, True])
@pytest.mark.parametrize('saturation_holds', [False, True])
def test_postcondition_never_overwrites_a_prior_rejection(
    acceptable, represented, saturation_holds,
):
    control = ChannelPCGControl(fixed_work=True)
    for reason in ('fixed_work_completed', 'fixed_work_roundoff_saturation'):
        result, _ = control.finish_fixed_work_postcondition(
            acceptable=acceptable, represented_at_floor=represented,
            termination_reason=reason,
            saturation_reason_still_at_floor=saturation_holds,
        )
        assert result is (
            acceptable and represented
            and (reason != 'fixed_work_roundoff_saturation' or saturation_holds)
        )


@pytest.mark.parametrize('path,iterations,error,message,termination', [
    ('production', 200, 'ChannelPressureConvergenceError',
     'production Channel pressure solve did not satisfy its contract',
     'fixed_work_completed'),
    ('transpose', 200, 'FunctionalConvergenceError',
     'standalone Channel pressure transpose solve is unacceptable',
     'fixed_work_completed'),
    *[('implicit', n, 'RuntimeError',
       'pressure-loop and diagnostics postconditions disagree', None)
      for n in (30, 40, 44)],
])
def test_failed_floor_retry_preserves_parent_path_exception(
    path, iterations, error, message, termination,
):
    command = [sys.executable, '-I',
               str(ROOT / 'benchmarks/probe_v0_4_s18_r5_postcondition.py'),
               str(ROOT), path, str(iterations)]
    run = subprocess.run(command, capture_output=True, text=True, check=True,
                         env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                              'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    report = json.loads(run.stdout)
    assert report['record']['outcome'] == 'returned'
    assert report['record']['termination'] == 'fixed_work_completed'
    injected = report['inject']
    assert injected['forced']['real_represented'] is True
    assert injected['equal_forced'] is not None
    assert injected['nan_injected'] is True
    assert injected['outcome'] == 'raised'
    assert injected['exc'] == error
    assert injected['message'] == message
    assert injected['termination'] == termination
