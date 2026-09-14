"""CPU tests for Stage N shadow-runtime performance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pytest
import torch

from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.experimental import (
    RuntimePerformanceRecorder,
    build_experimental_model_runtime,
    create_canary_geometry_solver_registry,
)
from pssolver.experimental.stage_n_diagnostics import (
    _operator_audit,
    analyze_stage_n_diagnostics,
    summarize_audited_operators,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.canary import DiffusionHelmholtzCouplingModel


P = PeriodicBC()
N = HomogeneousNeumannBC()
BOUNDARIES = BoundarySet((P, N))


def _runtime(*, instrumented: bool):
    model = DiffusionHelmholtzCouplingModel(
        boundaries=BOUNDARIES,
        diffusivity=0.07,
        coupling=-0.3,
        helmholtz_shift=1.1,
        helmholtz_length_sq=0.2,
        initial_amplitude=0.25,
        initial_modes=(1, 2),
    )
    numerics = NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
        ),
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=0,
    )
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        numerics,
        dt=0.01,
        batch_size=2,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
        enable_performance_instrumentation=instrumented,
    )


def test_runtime_instrumentation_is_opt_in_and_numerically_transparent():
    baseline = _runtime(instrumented=False)
    measured = _runtime(instrumented=True)
    assert baseline.performance_recorder is None
    recorder = measured.performance_recorder
    assert isinstance(recorder, RuntimePerformanceRecorder)
    recorder.reset()

    baseline.solver.run(3)
    measured.solver.run(3)

    assert torch.equal(
        baseline.solver.fields.spatial,
        measured.solver.fields.spatial,
    )
    assert torch.equal(
        baseline.solver.fields.spectral,
        measured.solver.fields.spectral,
    )
    report = recorder.snapshot()
    assert report["enabled"] is True
    assert report["device"] == "cpu"
    regions = report["regions"]
    assert regions["timestep.total"]["calls"] == 3
    assert regions["timestep.algebraic_update"]["calls"] == 3
    assert regions["algebraic.screened_response.solve"]["calls"] == 2
    assert regions["explicit_rhs.model"]["calls"] == 3
    assert regions["operators.forward_projected"]["calls"] > 0
    assert regions["transform.forward"]["calls"] > 0
    assert regions["transform.inverse"]["calls"] > 0
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_runtime_performance_recorder_requires_concrete_cuda_identity():
    with pytest.raises(ValueError, match="concrete device index"):
        RuntimePerformanceRecorder("cuda")


@dataclass
class _FakeEvent:
    key: str
    count: int
    self_cpu_memory_usage: int = 0
    self_device_memory_usage: int = 0


def test_operator_audit_keeps_only_frozen_allocation_and_movement_set():
    report = summarize_audited_operators(
        (
            _FakeEvent("aten::clone", 4, 10, 20),
            _FakeEvent("aten::empty", 7, 30, 40),
            _FakeEvent("aten::sin", 100, 50, 60),
        )
    )

    assert report["total_selected_calls"] == 11
    assert report["operators"]["aten::clone"] == {
        "calls": 4,
        "self_cpu_memory_bytes": 10,
        "self_device_memory_bytes": 20,
    }
    assert "aten::sin" not in report["operators"]


def test_operator_audit_executes_independently_on_cpu_canary():
    runtime = _runtime(instrumented=True)
    report = _operator_audit(runtime, 1)

    assert report["schema_version"] == 1
    assert report["total_selected_calls"] > 0
    assert report["operators"]["aten::stack"]["calls"] > 0


def _diagnostic(*, algebraic_fraction: float, rhs_fraction: float):
    regions = {}
    values = {
        "timestep.total": 1.0,
        "timestep.algebraic_update": algebraic_fraction,
        "timestep.explicit_rhs": rhs_fraction,
        "timestep.spectral_update": 0.04,
        "timestep.dynamic_projection": 0.02,
        "timestep.dynamic_inverse": 0.03,
        "timestep.spectral_refresh": 0.01,
        "algebraic.flow.solve": algebraic_fraction * 0.6,
        "transform.forward": 0.2,
        "transform.inverse": 0.3,
    }
    for name, fraction in values.items():
        regions[name] = {
            "calls": 20,
            "total_seconds": fraction,
            "mean_seconds_per_call": fraction / 20.0,
            "peak_allocated_bytes_observed": 100,
            "peak_reserved_bytes_observed": 200,
            "fraction_of_timestep_total": fraction,
        }
    operators = {
        name: {
            "calls": 4 if name == "aten::clone" else 0,
            "self_cpu_memory_bytes": 0,
            "self_device_memory_bytes": 0,
        }
        for name in (
            "aten::_to_copy",
            "aten::cat",
            "aten::clone",
            "aten::contiguous",
            "aten::copy_",
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::stack",
            "aten::to",
        )
    }
    return {
        "qualification_stage": "N",
        "classification": "DIAGNOSTIC_COMPLETE",
        "finite": True,
        "configuration": {"shape": [8, 7], "profile_steps": 20},
        "production_metadata_sha256": "a" * 64,
        "production_initial_q_sha256": "b" * 64,
        "throughput": {"mean_timestep_seconds": 0.01},
        "semantic_regions": {"enabled": True, "regions": regions},
        "operator_audit": {
            "steps": 2,
            "operators": operators,
        },
    }


def test_stage_n_analysis_ranks_regions_without_authorizing_promotion(tmp_path):
    paths = []
    for index, value in enumerate(((0.7, 0.2), (0.6, 0.3)), start=1):
        path = tmp_path / f"diagnostic_{index}.json"
        path.write_text(
            json.dumps(
                _diagnostic(
                    algebraic_fraction=value[0],
                    rhs_fraction=value[1],
                )
            ),
            encoding="utf-8",
        )
        paths.append(path)

    report = analyze_stage_n_diagnostics(paths)

    assert report["classification"] == "DIAGNOSTIC_COMPLETE"
    assert report["architecture_decision"] == (
        "retain_shadow_without_promotion"
    )
    assert report["eligible_for_stage_n1_optimization_design"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["ranked_top_level_regions"][:2] == [
        "timestep.algebraic_update",
        "timestep.explicit_rhs",
    ]
    assert report["ranked_algebraic_solvers"] == [
        "algebraic.flow.solve"
    ]
    assert report["operator_counts"]["aten::clone"][
        "mean_calls_per_step"
    ] == pytest.approx(2.0)
