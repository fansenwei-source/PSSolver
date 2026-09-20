"""Phase 3.2 bounded-workspace contracts."""

from __future__ import annotations

import pytest
import torch

import pssolver
import pssolver.execution as execution
from pssolver.execution.workspace import (
    RuntimeWorkspace,
    WorkspacePlan,
    WorkspaceSlotSpec,
)


def _plan(*, maximum_bytes: int | None = None) -> WorkspacePlan:
    return WorkspacePlan(
        slots=(
            WorkspaceSlotSpec(
                "molecular_field",
                (5, 1, 4, 3),
                torch.float64,
                "generation-local molecular field",
            ),
            WorkspaceSlotSpec(
                "force_spectral",
                (3, 1, 4, 2),
                torch.complex128,
                "generation-local projected force",
            ),
        ),
        device="cpu",
        maximum_bytes=maximum_bytes,
    )


def test_workspace_contract_is_provisional_and_disconnected():
    names = {"RuntimeWorkspace", "WorkspacePlan", "WorkspaceSlotSpec"}
    assert names.isdisjoint(execution.__all__)
    assert names.isdisjoint(pssolver.__all__)


def test_workspace_plan_is_bounded_and_has_stable_slot_handles():
    plan = _plan(maximum_bytes=4096)
    assert plan.slot_index("molecular_field") == 0
    assert plan.slot_index("force_spectral") == 1
    assert plan.required_bytes == 864
    assert plan.maximum_bytes == 4096
    metadata = plan.to_metadata()
    assert metadata["slot_count"] == 2
    assert metadata["required_bytes"] == 864
    assert [slot["name"] for slot in metadata["slots"]] == [
        "molecular_field",
        "force_spectral",
    ]
    with pytest.raises(KeyError, match="unknown workspace slot"):
        plan.slot_index("missing")


def test_workspace_allocates_once_and_invalidates_between_generations():
    workspace = _plan().allocate()
    assert isinstance(workspace, RuntimeWorkspace)
    token_1 = workspace.begin_generation()
    molecular = workspace.buffer("molecular_field", token=token_1)
    pointer = molecular.untyped_storage().data_ptr()
    molecular.fill_(3.0)
    with pytest.raises(RuntimeError, match="not valid"):
        workspace.require("molecular_field", token=token_1)
    workspace.publish("molecular_field", token=token_1)
    assert workspace.require("molecular_field", token=token_1) is molecular
    assert workspace.to_metadata()["published_slots"] == ["molecular_field"]
    workspace.end_generation(token=token_1)

    token_2 = workspace.begin_generation()
    assert token_2 == token_1 + 1
    molecular_2 = workspace.buffer_at(0, token=token_2)
    assert molecular_2 is molecular
    assert molecular_2.untyped_storage().data_ptr() == pointer
    with pytest.raises(RuntimeError, match="not valid"):
        workspace.require_at(0, token=token_2)
    with pytest.raises(RuntimeError, match="stale or inactive"):
        workspace.buffer_at(0, token=token_1)
    workspace.abort_generation(token=token_2)
    assert workspace.active is False


def test_workspace_generation_cannot_overlap_or_publish_out_of_range():
    workspace = _plan().allocate()
    token = workspace.begin_generation()
    with pytest.raises(RuntimeError, match="already active"):
        workspace.begin_generation()
    with pytest.raises(IndexError, match="out of range"):
        workspace.publish_at(2, token=token)
    workspace.end_generation(token=token)
    with pytest.raises(RuntimeError, match="stale or inactive"):
        workspace.end_generation(token=token)


def test_workspace_plan_rejects_duplicate_or_unbounded_allocations():
    slot = WorkspaceSlotSpec("gradient", (2, 3), torch.float32, "gradient")
    with pytest.raises(ValueError, match="unique"):
        WorkspacePlan(slots=(slot, slot), device="cpu")
    with pytest.raises(ValueError, match="exceeds"):
        _plan(maximum_bytes=800)


@pytest.mark.parametrize(
    "kwargs,exception,match",
    [
        (
            {
                "name": "not-a-name",
                "shape": (2,),
                "dtype": torch.float32,
                "purpose": "temporary",
            },
            ValueError,
            "identifier",
        ),
        (
            {
                "name": "temporary",
                "shape": (2, -1),
                "dtype": torch.float32,
                "purpose": "temporary",
            },
            ValueError,
            "shape",
        ),
        (
            {
                "name": "temporary",
                "shape": (2,),
                "dtype": "float32",
                "purpose": "temporary",
            },
            TypeError,
            "dtype",
        ),
    ],
)
def test_workspace_slot_validation(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        WorkspaceSlotSpec(**kwargs)
