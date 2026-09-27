"""P8.5.5 finite-Q anchoring workflow, restart, and package tests."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

import pssolver
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.finite_q_anchoring import (
    apply_plane_finite_q_anchoring_pilot,
    lower_plane_finite_q_anchoring_pilot,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    quadratic_finite_q_anchoring,
    uniaxial_Q,
)
from pssolver.runtime.finite_q_anchoring import (
    PlaneFiniteQAnchoringCheckpoint,
    PlaneFiniteQAnchoringRuntime,
)
from pssolver.workflows.finite_q_anchoring import (
    PlaneFiniteQAnchoringWorkflow,
    load_finite_q_anchoring_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _construction(
    tmp_path: Path,
    *,
    lower_strength: float = 0.04,
    upper_strength: float = 0.07,
    seed: int = 855,
):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=Path("p855_unused_output"),
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=6,
        ny=5,
        nz=6,
        lx=12.0,
        ly=10.0,
        height=20.0,
        steps=4,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    anchoring = quadratic_finite_q_anchoring(
        k_q=base.equation_system.parameters["ldg_l1"],
        wall_strengths={
            (2, "lower"): lower_strength,
            (2, "upper"): upper_strength,
        },
        target_q={
            (2, "lower"): uniaxial_Q(
                np.asarray((0.0, 0.0, 1.0)),
                0.6,
            ),
            (2, "upper"): uniaxial_Q(
                np.asarray((1.0, 0.0, 0.0)),
                0.5,
            ),
        },
    )
    simulation = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    plan = lower_plane_finite_q_anchoring_pilot(simulation, anchoring)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = {
        component: 0.03
        * torch.randn(
            plan.component_plans[0].domain_shape,
            generator=generator,
            dtype=torch.float64,
        )
        for component in Q_COMPONENTS
    }
    return plan, initial


def _runtime(tmp_path: Path, **kwargs) -> PlaneFiniteQAnchoringRuntime:
    plan, initial = _construction(tmp_path, **kwargs)
    return PlaneFiniteQAnchoringRuntime(plan, initial, dt=0.005)


def _snapshot(runtime: PlaneFiniteQAnchoringRuntime):
    return {
        component: (
            scalar.remainder.clone(),
            scalar.bounded_modal.clone(),
        )
        for component, scalar in runtime.components.items()
    }, runtime.completed_steps


def _assert_snapshot(runtime, snapshot):
    tensors, completed_steps = snapshot
    assert runtime.completed_steps == completed_steps
    for component, (remainder, modal) in tensors.items():
        assert torch.equal(runtime.components[component].remainder, remainder)
        assert torch.equal(runtime.components[component].bounded_modal, modal)


def test_five_component_relaxation_is_finite_oriented_and_explicitly_partial(
    tmp_path,
):
    runtime = _runtime(tmp_path)
    before = runtime.physical_q().clone()
    runtime.advance(3)
    after = runtime.physical_q()
    assert runtime.completed_steps == 3
    assert after.shape == (6, 5, 6, 5)
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
    for lower, upper in runtime.boundary_residuals().values():
        torch.testing.assert_close(lower, torch.zeros_like(lower), atol=1e-14, rtol=0)
        torch.testing.assert_close(upper, torch.zeros_like(upper), atol=1e-14, rtol=0)
    metadata = runtime.to_metadata()
    assert metadata["complete_q_timestep_connected"] is False
    assert metadata["public_runner_connected"] is False
    assert metadata["operator_cache_entry_count"] == 5
    assert runtime.checkpoint_identity_metadata()["evolution_law"] == (
        "implicit_wall_normal_elastic_relaxation"
    )


def test_continuous_and_in_memory_split_restart_are_byte_identical(tmp_path):
    continuous = _runtime(tmp_path / "continuous")
    segment = _runtime(tmp_path / "segment")
    resumed = _runtime(tmp_path / "resumed")
    continuous.advance(4)
    segment.advance(2)
    checkpoint = segment.capture_checkpoint()
    resumed.restore_checkpoint(checkpoint)
    resumed.advance(2)
    assert resumed.completed_steps == 4
    assert torch.equal(continuous.physical_q(), resumed.physical_q())
    for component in Q_COMPONENTS:
        assert torch.equal(
            continuous.components[component].remainder,
            resumed.components[component].remainder,
        )
        assert torch.equal(
            continuous.components[component].bounded_modal,
            resumed.components[component].bounded_modal,
        )


def test_file_workflow_restart_and_saved_physical_q_are_exact(tmp_path):
    continuous = _runtime(tmp_path / "continuous_runtime")
    segment = _runtime(tmp_path / "segment_runtime")
    resumed = _runtime(tmp_path / "resumed_runtime")
    continuous_result = PlaneFiniteQAnchoringWorkflow(
        continuous,
        tmp_path / "continuous_output",
        save_interval=2,
        checkpoint_interval=2,
    ).run(4)
    segment_result = PlaneFiniteQAnchoringWorkflow(
        segment,
        tmp_path / "segment_output",
        save_interval=1,
        checkpoint_interval=2,
    ).run(2)
    restart_path = tmp_path / "segment_output" / "checkpoint_2"
    loaded = load_finite_q_anchoring_checkpoint(restart_path)
    assert loaded.completed_steps == 2
    resumed_result = PlaneFiniteQAnchoringWorkflow(
        resumed,
        tmp_path / "resumed_output",
        save_interval=2,
        checkpoint_interval=2,
    ).run(4, restart_from=restart_path)

    assert segment_result.final_step == 2
    assert resumed_result.start_step == 2
    assert torch.equal(continuous_result.final_q, resumed_result.final_q)
    assert np.array_equal(
        np.load(tmp_path / "continuous_output" / "Q_4.npy"),
        np.load(tmp_path / "resumed_output" / "Q_4.npy"),
    )
    assert json.loads(
        (tmp_path / "resumed_output" / "COMPLETE").read_text()
    )["finite"] is True


def _tamper_path(value, path):
    result = copy.deepcopy(value)
    target = result
    for key in path[:-1]:
        target = target[key]
    key = path[-1]
    current = target[key]
    if isinstance(current, str):
        target[key] = "tampered_identity"
    elif isinstance(current, (int, float)):
        target[key] = current + 1
    elif isinstance(current, list):
        target[key] = list(reversed(current))
    elif isinstance(current, dict):
        first = next(iter(current))
        target[key][first] = current[first] + 1.0
    else:
        raise AssertionError(f"unsupported tamper value: {current!r}")
    return result


@pytest.mark.parametrize(
    "path",
    (
        ("component_identities", "Qxx", "raw_lower_coefficients_sha256"),
        ("normal_derivative_convention",),
        ("faces", 0, "target_components"),
        ("q_convention", "id"),
        ("surface_law_id",),
        ("component_identities", "Qxx", "bounded_axis_operator_kind"),
        ("component_identities", "Qxx", "bounded_axis_plan_sha256"),
        ("component_identities", "Qxx", "geometry_name"),
        ("dtype",),
        ("runtime_path",),
        ("backend",),
    ),
)
def test_identity_tampering_is_rejected_before_any_component_mutation(
    tmp_path,
    path,
):
    source = _runtime(tmp_path / "source")
    target = _runtime(tmp_path / "target")
    source.advance(1)
    checkpoint = source.capture_checkpoint()
    identity = json.loads(checkpoint.identity_json)
    tampered = replace(
        checkpoint,
        identity_json=json.dumps(
            _tamper_path(identity, path),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    before = _snapshot(target)
    with pytest.raises(ValueError, match="runtime identity"):
        target.restore_checkpoint(tampered)
    _assert_snapshot(target, before)


def test_component_payload_tamper_is_rejected_atomically(tmp_path):
    source = _runtime(tmp_path / "source")
    target = _runtime(tmp_path / "target")
    source.advance(1)
    checkpoint = source.capture_checkpoint()
    components = list(checkpoint.components)
    name, scalar = components[-1]
    corrupted = scalar.remainder.clone()
    corrupted[0, 0, 0] += 1.0
    components[-1] = (name, replace(scalar, remainder=corrupted))
    tampered = replace(checkpoint, components=tuple(components))
    before = _snapshot(target)
    with pytest.raises(ValueError, match="payload identity"):
        target.restore_checkpoint(tampered)
    _assert_snapshot(target, before)


def test_file_payload_and_file_record_tampering_are_rejected(tmp_path):
    runtime = _runtime(tmp_path / "runtime")
    output = tmp_path / "output"
    PlaneFiniteQAnchoringWorkflow(
        runtime,
        output,
        checkpoint_interval=1,
    ).run(1)
    original = output / "checkpoint_1"

    payload_case = tmp_path / "payload_tamper"
    shutil.copytree(original, payload_case)
    tensor_path = payload_case / "Qxx.remainder.npy"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        load_finite_q_anchoring_checkpoint(payload_case)

    record_case = tmp_path / "record_tamper"
    shutil.copytree(original, record_case)
    metadata_path = record_case / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["components"]["Qxx"]["tensors"]["remainder"]["file"][
        "shape"
    ][0] += 1
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))
    with pytest.raises(ValueError, match="shape metadata"):
        load_finite_q_anchoring_checkpoint(record_case)


def test_neumann_limit_runs_without_erasing_target_provenance(tmp_path):
    runtime = _runtime(
        tmp_path,
        lower_strength=0.0,
        upper_strength=0.0,
    )
    identity = runtime.checkpoint_identity_metadata()
    assert all(face["wall_strength"] == 0.0 for face in identity["faces"])
    assert any(
        any(value != 0.0 for value in face["target_components"])
        for face in identity["faces"]
    )
    runtime.advance(2)
    assert torch.isfinite(runtime.physical_q()).all()


def test_wheel_contains_and_imports_finite_q_runtime_and_workflow(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        assert "pssolver/runtime/finite_q_anchoring.py" in names
        assert "pssolver/workflows/finite_q_anchoring.py" in names
        archive.extractall(installed)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib; "
                "import pssolver.runtime.finite_q_anchoring as r; "
                "import pssolver.workflows.finite_q_anchoring as w; "
                f"root=pathlib.Path({str(installed)!r}).resolve(); "
                "assert pathlib.Path(r.__file__).resolve().is_relative_to(root); "
                "assert pathlib.Path(w.__file__).resolve().is_relative_to(root); "
                "print(r.FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION, "
                "w.FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "1 1"


def test_public_api_and_production_robin_rejection_remain_unchanged():
    assert not hasattr(pssolver, "PlaneFiniteQAnchoringRuntime")
    assert not hasattr(pssolver, "PlaneFiniteQAnchoringWorkflow")
    assert not hasattr(pssolver, "quadratic_finite_q_anchoring")


def test_p855_record_and_archive_entries_are_present():
    record = json.loads(
        (NOTES / "phase_8_p855_finite_q_workflow_restart.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["phase"] == "P8.5.5"
    assert record["classification"] == (
        "PASS_P8_5_5_FINITE_Q_WORKFLOW_RESTART_CPU_CLOSURE"
    )
    assert record["complete_q_timestep_connected"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["eligible_for_p8_5_6"] is True
    assert record["authorization"]["p8_5_6_implemented"] is False
    archive = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_8_p855_finite_q_workflow_restart.md",
        "phase_8_p855_finite_q_workflow_restart.json",
    ):
        assert archive.count(f'"{name}"') == 1
