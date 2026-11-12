"""Failure contracts remain identical after a failed fixed-work floor retry."""

from pathlib import Path

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
