"""CPU contracts for Stage Q.6.2 native-segment handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    ProjectedBatchAssemblyMode,
    ProjectedBatchAssemblyPolicy,
    ProjectedTransformDirection,
    analyze_stage_q62_qualification,
    build_stage_q62_h100_plan,
)


PROJECT_ROOT = Path(__file__).parents[1]
REQUIRED_SOURCES = (
    "algebraic.nematic_stress.dependencies",
    "explicit_rhs.dependencies",
    "explicit_rhs.outputs",
)


class _TransformContext:
    batch_size = 1
    physical_shape = (2, 3)
    spectral_shape = (2, 3)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.packed_calls: list[int] = []

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
        return value.real.clone()

    def transform_projected_packed(
        self,
        boundaries: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        assert boundaries == ("periodic", "neumann")
        self.packed_calls.append(int(value.shape[0]))
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real.clone()


def _values() -> tuple[torch.Tensor, ...]:
    first = torch.arange(12, dtype=torch.float64).reshape(2, 1, 2, 3)
    middle = torch.full((1, 2, 3), 17.0, dtype=torch.float64)
    last = torch.arange(12, 24, dtype=torch.float64).reshape(2, 1, 2, 3)
    return (first[0], first[1], middle, last[0], last[1])


def test_native_segments_preserve_order_without_copy_or_workspace():
    context = _TransformContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=ProjectedBatchAssemblyPolicy.native_segments(
            source_names=REQUIRED_SOURCES
        ),
        enable_batch_assembly_diagnostics=True,
    )
    values = _values()
    result = scheduler.forward_values_many(
        tuple(f"q{index}" for index in range(len(values))),
        values,
        attribution_source=REQUIRED_SOURCES[0],
    )

    assert context.packed_calls == [2, 1, 2]
    for actual, expected in zip(result, values, strict=True):
        assert torch.equal(actual.real, expected)
    diagnostics = scheduler.batch_assembly_diagnostics()
    source = diagnostics["source_attribution"][REQUIRED_SOURCES[0]]
    assert source["native_segment_groups"] == 1
    assert source["native_segment_batches"] == 3
    assert source["native_segment_components"] == 5
    assert source["native_segment_singleton_batches"] == 1
    assert source["native_segment_extra_transform_batches"] == 2
    assert source["native_segment_materialized_output_bytes"] == 0
    assert source["copy_cat_batches"] == 0
    assert source["fallback_reasons"] == {}
    assert diagnostics["workspace_count"] == 0
    assert diagnostics["retained_tensor_references"] == 0


def test_native_segment_policy_is_exact_source_scoped():
    context = _TransformContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=ProjectedBatchAssemblyPolicy.native_segments(
            source_names=REQUIRED_SOURCES
        ),
        enable_batch_assembly_diagnostics=True,
    )
    scheduler.forward_values_many(
        ("a", "b"),
        _values()[:2],
        attribution_source="algebraic.other.dependencies",
    )
    source = scheduler.batch_assembly_diagnostics()["source_attribution"][
        "algebraic.other.dependencies"
    ]
    assert source["native_segment_batches"] == 0
    assert source["copy_cat_batches"] == 1
    assert source["fallback_reasons"] == {"policy_copy_cat": 1}


def test_native_segment_policy_is_fail_closed_and_self_describing():
    policy = ProjectedBatchAssemblyPolicy.native_segments(
        source_names=REQUIRED_SOURCES
    )
    assert policy.mode is ProjectedBatchAssemblyMode.NATIVE_SEGMENTS
    assert policy.use_native_segments is True
    assert policy.use_preallocated_workspace is False
    assert policy.allow_contiguous_storage_view is False
    metadata = policy.to_metadata()
    assert metadata["native_segment_sources"] == list(REQUIRED_SOURCES)
    assert metadata["fallback"] == "fail_closed_for_designated_sources"
    assert metadata["changes_transform_partition"] is True
    assert metadata["owns_runtime_workspace"] is False

    with pytest.raises(ValueError, match="at least one exact source"):
        ProjectedBatchAssemblyPolicy.native_segments(source_names=())
    with pytest.raises(ValueError, match="unique"):
        ProjectedBatchAssemblyPolicy.native_segments(
            source_names=(REQUIRED_SOURCES[0], REQUIRED_SOURCES[0])
        )


def test_stage_q62_runtime_does_not_enter_production_plane_or_channel():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "Channel.py",
        "pssolver/solver.py",
        "pssolver/geometries/tensor_product.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_q62" not in text.lower()
        assert "native_segments" not in text


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q61_report(path: Path) -> str:
    _write_json(
        path,
        {
            "qualification_stage": "Q.6.1",
            "classification": "NATIVE_HANDOFF_CONTRACT_COMPLETE",
            "architecture_decision": (
                "producer_owned_boundary_signature_native_segments"
            ),
            "eligible_for_stage_q62_candidate_implementation": True,
            "eligible_for_stage_q62_production_promotion": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "stage_q62_candidate": {
                "name": "plane_native_segment_handoff_shadow_candidate",
                "scope": "experimental_plane_shadow_runtime_only",
                "required_sources": list(REQUIRED_SOURCES),
                "fallback_policy": "fail_closed_no_copy_cat_fallback",
            },
        },
    )
    return _sha256(path)


def _compiled_rhs() -> dict[str, object]:
    return {
        "owner": "geometry_executor",
        "implementation_name": "plane_beris_edwards_pointwise_explicit_rhs",
        "observability": {
            "pointwise_kernels": {
                "requested": "compile",
                "effective": "compile",
                "compile": {
                    "backend": "inductor",
                    "fullgraph": True,
                    "dynamic": False,
                },
                "fallback_allowed": False,
                "fallback_reason": None,
            },
            "fallback_to_model": False,
        },
    }


def _producer_outputs() -> dict[str, object]:
    output = {
        "producer_output_layout": "boundary_packed",
        "producer_packing": {
            "ownership": "producer",
            "packing_site": "inside_pointwise_kernel",
            "pointwise_execution": "compile",
            "post_kernel_stack": False,
            "post_kernel_cat": False,
            "cross_generation_reuse": False,
        },
    }
    return {"molecular_field": output, "nematic_stress": output}


def _diagnostics(role: str) -> dict[str, object]:
    sources = {}
    for source in REQUIRED_SOURCES:
        counters = {
            "copy_cat_batches": 4 if role == "baseline" else 0,
            "copy_cat_materialized_output_bytes": (
                4096 if role == "baseline" else 0
            ),
            "native_segment_groups": 0 if role == "baseline" else 4,
            "native_segment_batches": 0 if role == "baseline" else 8,
            "native_segment_components": 0 if role == "baseline" else 12,
            "native_segment_logical_input_bytes": (
                0 if role == "baseline" else 4096
            ),
            "native_segment_materialized_output_bytes": 0,
            "native_segment_singleton_batches": (
                0 if role == "baseline" else 4
            ),
            "native_segment_extra_transform_batches": (
                0 if role == "baseline" else 4
            ),
            "fallback_reasons": {},
        }
        sources[source] = counters
    return {
        "enabled": True,
        "policy": {
            "mode": (
                "contiguous_storage_view"
                if role == "baseline"
                else "native_segments"
            )
        },
        "source_attribution": sources,
        "workspace_count": 0,
        "workspace_allocated_bytes": 0,
        "workspace_active_count": 0,
        "workspace_retains_timestep_inputs": False,
        "retained_tensor_references": 0,
    }


def _trajectory_diagnostics(role: str) -> dict[str, object]:
    diagnostics = _diagnostics(role)
    diagnostics["enabled"] = False
    diagnostics["source_attribution"] = {}
    return diagnostics


def _trajectory(root: Path, role: str) -> None:
    root.mkdir(parents=True)
    (root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    for step in range(7):
        for field in ("Q", "u", "p"):
            value = np.full((2, 2), step + len(field), dtype=np.float64)
            if role == "candidate" and step > 0:
                value = value + 1.0e-14
            np.save(root / f"{field}_{step}.npy", value)
    _write_json(
        root / "stage_q62_metrics.json",
        {
            "qualification_stage": "Q.6.2",
            "classification": "TRAJECTORY_COMPLETE",
            "measurement_role": role,
            "completed_steps": 6,
            "saved_steps": list(range(7)),
            "projected_batch_assembly_policy": {
                "mode": (
                    "contiguous_storage_view"
                    if role == "baseline"
                    else "native_segments"
                )
            },
            "production_default_changed": False,
            "production_metadata_sha256": "metadata",
            "production_initial_q_sha256": "q0",
            "explicit_rhs_execution": _compiled_rhs(),
            "producer_outputs": _producer_outputs(),
            "batch_assembly_diagnostics": _trajectory_diagnostics(role),
        },
    )


def _profile(path: Path, role: str, seconds: float) -> None:
    _write_json(
        path,
        {
            "qualification_stage": "Q.6.2",
            "classification": "PROFILE_COMPLETE",
            "measurement_role": role,
            "configuration": {
                "shape": [320, 320, 80],
                "warmup_steps": 10,
                "profile_steps": 20,
            },
            "projected_batch_assembly_policy": {
                "mode": (
                    "contiguous_storage_view"
                    if role == "baseline"
                    else "native_segments"
                )
            },
            "production_default_changed": False,
            "production_metadata_sha256": "metadata",
            "production_initial_q_sha256": "q0",
            "finite": True,
            "environment": {
                "device_name": "NVIDIA H100 PCIe",
                "cuda_matmul_allow_tf32": False,
            },
            "explicit_rhs_execution": _compiled_rhs(),
            "producer_outputs": _producer_outputs(),
            "batch_assembly_diagnostics": _diagnostics(role),
            "throughput": {"mean_timestep_seconds": seconds},
            "memory": {
                "peak_allocated_bytes": (
                    1000 if role == "baseline" else 900
                ),
                "peak_reserved_bytes": (
                    2000 if role == "baseline" else 1800
                ),
            },
        },
    )


def test_stage_q62_analyzer_accepts_three_source_zero_copy_candidate(tmp_path):
    evidence = tmp_path / "q61.json"
    digest = _q61_report(evidence)
    trajectories = {
        role: tmp_path / f"trajectory_{role}"
        for role in ("baseline", "candidate")
    }
    for role, path in trajectories.items():
        _trajectory(path, role)
    profiles = {role: [] for role in ("baseline", "candidate")}
    for role in profiles:
        for trial in range(3):
            path = tmp_path / f"{role}_{trial}.json"
            _profile(path, role, 1.0 if role == "baseline" else 0.8)
            profiles[role].append(path)

    report = analyze_stage_q62_qualification(
        trajectories["baseline"],
        trajectories["candidate"],
        profiles["baseline"],
        profiles["candidate"],
        stage_q61_report=evidence,
        expected_stage_q61_sha256=digest,
    )

    assert report["classification"] == "A_recommended"
    assert report["gates"] == {
        "numerical_equivalence": True,
        "three_source_copy_elimination": True,
        "r320_performance_improvement": True,
        "r320_memory_non_regression": True,
        "candidate_safety_non_regression": True,
    }
    assert report["eligible_for_stage_q63_architecture_decision"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["trajectory"]["maximum_relative_l2"] < 1.0e-12
    assert len(report["native_segment_fragmentation"]) == 3


def test_stage_q62_analyzer_rejects_copy_at_designated_candidate_profile(
    tmp_path,
):
    evidence = tmp_path / "q61.json"
    digest = _q61_report(evidence)
    trajectories = {
        role: tmp_path / f"trajectory_{role}"
        for role in ("baseline", "candidate")
    }
    for role, path in trajectories.items():
        _trajectory(path, role)
    profiles = {role: [] for role in ("baseline", "candidate")}
    for role in profiles:
        for trial in range(3):
            path = tmp_path / f"{role}_{trial}.json"
            _profile(path, role, 1.0)
            profiles[role].append(path)
    candidate_profile = profiles["candidate"][0]
    report = json.loads(candidate_profile.read_text(encoding="utf-8"))
    source = report["batch_assembly_diagnostics"]["source_attribution"][
        REQUIRED_SOURCES[1]
    ]
    source["copy_cat_batches"] = 1
    _write_json(candidate_profile, report)

    with pytest.raises(ValueError, match="profile contract differs"):
        analyze_stage_q62_qualification(
            trajectories["baseline"],
            trajectories["candidate"],
            profiles["baseline"],
            profiles["candidate"],
            stage_q61_report=evidence,
            expected_stage_q61_sha256=digest,
        )


def test_stage_q62_formally_rejects_fragmented_candidate_on_safety_ratio(
    tmp_path,
):
    evidence = tmp_path / "q61.json"
    digest = _q61_report(evidence)
    trajectories = {
        role: tmp_path / f"trajectory_{role}"
        for role in ("baseline", "candidate")
    }
    for role, path in trajectories.items():
        _trajectory(path, role)
    profiles = {role: [] for role in ("baseline", "candidate")}
    for role in profiles:
        for trial in range(3):
            path = tmp_path / f"{role}_{trial}.json"
            _profile(path, role, 1.0 if role == "baseline" else 1.16)
            profiles[role].append(path)

    report = analyze_stage_q62_qualification(
        trajectories["baseline"],
        trajectories["candidate"],
        profiles["baseline"],
        profiles["candidate"],
        stage_q61_report=evidence,
        expected_stage_q61_sha256=digest,
    )

    assert report["classification"] == "C_rejected"
    assert report["gates"]["numerical_equivalence"] is True
    assert report["gates"]["three_source_copy_elimination"] is True
    assert report["gates"]["candidate_safety_non_regression"] is False
    assert report["eligible_for_stage_q63_architecture_decision"] is False


def test_stage_q62_plan_is_balanced_and_analysis_last(tmp_path):
    evidence = tmp_path / "q61.json"
    digest = _q61_report(evidence)
    reference = tmp_path / "reference"
    reference.mkdir()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = build_stage_q62_h100_plan(
        project_root=PROJECT_ROOT,
        production_reference=reference,
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        python=Path(sys.executable),
        expected_commit=head,
        stage_q61_report=evidence,
        expected_stage_q61_sha256=digest,
    )
    profiles = plan["commands"]["profiles_balanced"]
    assert [value["role"] for value in profiles] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert len(plan["commands"]["trajectories"]) == 2
    assert plan["commands"]["analysis"][-2:] == [
        "--output",
        str(tmp_path / "control" / "stage_q62_qualification.json"),
    ]
    assert plan["fixed_contract"]["changes_production_default"] is False
