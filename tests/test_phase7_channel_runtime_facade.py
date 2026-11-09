"""P7.4 opt-in Channel runtime, output, checkpoint, and restart contracts."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


NOTES = ROOT / "notes" / "architecture_v0_2"


SHAPE = (8, 6, 5)


LENGTHS = (8.0, 6.0, 5.0)


def _state_clone(adapter):
    return (
        adapter.fields.spatial.detach().clone(),
        adapter.fields.spectral.detach().clone(),
        adapter.completed_steps,
    )


def _assert_state_unchanged(adapter, before):
    physical, spectral, completed = before
    assert torch.equal(adapter.fields.spatial, physical)
    assert torch.equal(adapter.fields.spectral, spectral)
    assert adapter.completed_steps == completed
