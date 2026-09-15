"""CPU contracts for Stage O.4.3.4 materialization attribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    ProjectedBatchAssemblyPolicy,
    ProjectedTransformDirection,
    RuntimePerformanceRecorder,
)


PROJECT_ROOT = Path(__file__).parents[1]


class _BatchContext:
    batch_size = 1
    physical_shape = (2, 3)
    spectral_shape = (2, 3)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def boundary_conditions(self, name: str) -> tuple[str, ...]:
        del name
        return ("periodic", "neumann")

    def forward_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return torch.complex(value, torch.zeros_like(value))

    def inverse_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return value.real

    def transform_projected_packed(
        self,
        boundaries: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        assert boundaries == ("periodic", "neumann")
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real


def _scheduler() -> tuple[
    BoundarySignatureTransformScheduler,
    RuntimePerformanceRecorder,
]:
    recorder = RuntimePerformanceRecorder(torch.device("cpu"))
    scheduler = BoundarySignatureTransformScheduler(
        _BatchContext(),
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        enable_batch_assembly_diagnostics=True,
        performance_recorder=recorder,
    )
    return scheduler, recorder


def test_scheduler_attributes_copy_bytes_and_timing_without_tensor_retention():
    scheduler, recorder = _scheduler()
    values = tuple(
        torch.full((1, 2, 3), float(index), dtype=torch.float64)
        for index in range(3)
    )
    with scheduler.attribution_scope("explicit_rhs.outputs"):
        scheduler.forward_values_many(("a", "b", "c"), values)

    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["schema_version"] == 2
    assert diagnostics["copy_cat_batches"] == 1
    assert diagnostics["copy_cat_materialized_output_bytes"] == 144
    source = diagnostics["source_attribution"]["explicit_rhs.outputs"]
    assert source["copy_cat_batches"] == 1
    assert source["copy_cat_components"] == 3
    assert source["copy_cat_materialized_output_bytes"] == 144
    assert source["contiguous_view_materialized_output_bytes"] == 0
    assert source["retained_tensor_references"] == 0
    assert json.dumps(dict(diagnostics), allow_nan=False, sort_keys=True)

    regions = recorder.snapshot()["regions"]
    region = source["timing_region"]
    assert regions[region]["calls"] == 1
    assert regions[region]["total_seconds"] >= 0.0


def test_scheduler_attributes_contiguous_views_and_reset_is_complete():
    scheduler, _ = _scheduler()
    packed = torch.arange(18, dtype=torch.float64).reshape(3, 1, 2, 3)
    with scheduler.attribution_scope("algebraic.nematic_stress.outputs"):
        scheduler.forward_values_many(
            ("a", "b", "c"),
            tuple(packed[index] for index in range(3)),
        )
    source = scheduler.batch_assembly_diagnostics()["source_attribution"][
        "algebraic.nematic_stress.outputs"
    ]
    assert source["contiguous_view_batches"] == 1
    assert source["contiguous_view_components"] == 3
    assert source["contiguous_view_materialized_output_bytes"] == 0
    assert source["copy_cat_materialized_output_bytes"] == 0

    scheduler.reset_batch_assembly_diagnostics()
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["source_attribution"] == {}
    assert diagnostics["copy_cat_batches"] == 0
    assert diagnostics["contiguous_view_batches"] == 0


def test_explicit_source_overrides_scope_and_invalid_sources_fail():
    scheduler, _ = _scheduler()
    values = tuple(
        torch.ones((1, 2, 3), dtype=torch.float64) for _ in range(2)
    )
    with scheduler.attribution_scope("outer.source"):
        scheduler.forward_values_many(
            ("a", "b"),
            values,
            attribution_source="explicit.override",
        )
    assert set(
        scheduler.batch_assembly_diagnostics()["source_attribution"]
    ) == {"explicit.override"}
    with pytest.raises(ValueError, match="attribution source"):
        with scheduler.attribution_scope("contains spaces"):
            pass


def test_stage_o434_instrumentation_remains_experimental_and_plane_only():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_o434" not in source
        assert "source_attribution" not in source
