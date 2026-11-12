"""P7.7.10 common application runner and public result protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import pssolver
from pssolver import (
    Simulation,
    SimulationDiagnosticProtocol,
    SimulationObservationProtocol,
    SimulationResult,
    SimulationRunStatus,
    compile_simulation,
    run_simulation,
)
from pssolver.functional.legacy_declarations.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)


ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p7710_public_runner_result.json"
)


REVIEWED_SOURCES = (
    "pssolver/api/results.py",
    "pssolver/api/runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _application_result(final_step: int = 2):
    observation = SimpleNamespace(
        step=final_step,
        q=np.zeros((16, 8, 8, 5), dtype=np.float32),
        velocity=np.zeros((16, 8, 8, 3), dtype=np.float32),
        pressure=np.zeros((16, 8, 8), dtype=np.float32),
    )
    diagnostic = SimpleNamespace(step=final_step)
    return SimpleNamespace(
        start_step=0,
        final_step=final_step,
        elapsed_seconds=1.5,
        saved_steps=(0, final_step),
        checkpoint_steps=(),
        final_observation=observation,
        diagnostics=(diagnostic,),
    )


def test_public_result_is_immutable_and_does_not_copy_observation_arrays():
    raw = _application_result()
    q = raw.final_observation.q
    result = SimulationResult(
        status=SimulationRunStatus.COMPLETE,
        application="test_application",
        runtime_path="test_runtime",
        source_simulation_sha256="a" * 64,
        application_request_sha256="b" * 64,
        output_directory=Path("data/result"),
        start_step=raw.start_step,
        final_step=raw.final_step,
        elapsed_seconds=raw.elapsed_seconds,
        saved_steps=raw.saved_steps,
        checkpoint_steps=raw.checkpoint_steps,
        final_observation=raw.final_observation,
        diagnostics=raw.diagnostics,
    )

    assert result.final_observation.q is q
    with pytest.raises(FrozenInstanceError):
        result.final_step = 3
    with pytest.raises(TypeError):
        result.provenance["changed"] = True


def test_dry_run_result_has_no_false_execution_outputs():
    result = SimulationResult(
        status=SimulationRunStatus.DRY_RUN,
        application="plane_complete_stress_beris_edwards",
        runtime_path="compiled_v2",
        source_simulation_sha256="a" * 64,
        application_request_sha256="b" * 64,
        output_directory=None,
        start_step=None,
        final_step=None,
        elapsed_seconds=None,
    )

    assert result.completed is False
    assert result.to_metadata()["final_observation"] is None


def test_public_runner_keeps_application_imports_lazy():
    path = ROOT / "pssolver/api/runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {
        name
        for name in top_level
        if name.startswith("pssolver.applications")
        or name.startswith("pssolver.runtime")
        or name.startswith("pssolver.workflows")
    }


def test_root_exports_public_result_contract():
    expected = {
        "SimulationDiagnosticProtocol",
        "SimulationObservationProtocol",
        "SimulationResult",
        "SimulationRunStatus",
    }
    assert expected <= set(pssolver.__all__)


def test_machine_record_binds_p7710_scope_and_source_identity():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P7_7_10_COMMON_RUNNER_AND_RESULT_PROTOCOL"
    )
    assert record["qualified_applications"] == [
        "plane_complete_stress_beris_edwards",
        "channel_legacy_active_force_active_nematics",
    ]
    assert record["p7_7_11_complete"] is False
    assert record["phase_8_authorized"] is False
    assert set(record["source_sha256"]) == set(REVIEWED_SOURCES)
    assert all(len(value) == 64 for value in record["source_sha256"].values())


def _plane_simulation(tmp_path, *, runtime_path="compiled_v2"):
    request = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "generated",
        runtime_path=runtime_path,
        device="cpu",
        dtype="float32",
        nx=16,
        ny=8,
        nz=8,
        steps=2,
        save_start_step=0,
        pointwise_execution="eager",
    )
    source = compose_plane_beris_edwards_simulation(decompose_plane_beris_edwards_run_spec(request))
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=replace(source.time_integration, refresh={"mode": "disabled"}),
        initial_condition=source.initial_condition,
        execution=replace(
            source.execution, options={**source.execution.options, "fallback_allowed": False}
        ),
        output=source.workflow,
        invocation=source.invocation,
    )


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
def test_run_simulation_dispatches_plane_once_and_returns_public_result(
    monkeypatch,
    tmp_path,
    runtime_path,
):
    simulation = _plane_simulation(tmp_path, runtime_path=runtime_path)
    compiled = compile_simulation(simulation)
    observed = {}
    fake = ModuleType("pssolver.runtime.legacy_entry")

    def run_plane_beris_edwards(
        run_spec,
        *,
        progress,
        public_compilation_metadata,
        emit_metadata,
    ):
        assert emit_metadata is False
        observed.update(
            run_spec=run_spec,
            progress=progress,
            public_compilation_metadata=public_compilation_metadata,
        )
        return _application_result()

    fake.run_v04_plane_translation = run_plane_beris_edwards
    monkeypatch.setitem(
        sys.modules,
        "pssolver.runtime.legacy_entry",
        fake,
    )
    progress = (0, 1)

    result = run_simulation(compiled, progress=progress)

    assert isinstance(result, SimulationResult)
    assert result.status is SimulationRunStatus.COMPLETE
    assert result.completed is True
    assert result.application == ("plane_complete_stress_beris_edwards")
    assert result.runtime_path == runtime_path
    assert result.output_directory == (tmp_path / "generated").resolve()
    assert result.start_step == 0
    assert result.final_step == 2
    assert result.saved_steps == (0, 2)
    assert isinstance(
        result.final_observation,
        SimulationObservationProtocol,
    )
    assert isinstance(result.diagnostics[0], SimulationDiagnosticProtocol)
    assert observed == {
        "run_spec": compiled.application_request,
        "progress": progress,
        "public_compilation_metadata": {
            "schema_version": 1,
            "compilation_identity_sha256": (compiled.compilation_identity_sha256),
            "source_simulation_sha256": compiled.source.canonical_sha256(),
            "application": compiled.application,
            "application_simulation_sha256": (
                compiled.application_specification.canonical_sha256()
            ),
            "application_request_sha256": (compiled.application_request.canonical_sha256()),
            "normalization": dict(compiled.normalization),
            "fallback_allowed": False,
        },
    }
    metadata = result.to_metadata()
    assert metadata["final_observation"]["q_shape"] == [16, 8, 8, 5]
    assert metadata["diagnostic_steps"] == [2]
    assert metadata["provenance"]["fallback_allowed"] is False
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_run_simulation_compiles_public_declaration_before_dispatch(
    monkeypatch,
    tmp_path,
):
    simulation = _plane_simulation(tmp_path)
    fake = ModuleType("pssolver.runtime.legacy_entry")
    calls = []

    def run_plane_beris_edwards(
        run_spec,
        *,
        progress,
        public_compilation_metadata,
        emit_metadata,
    ):
        assert emit_metadata is False
        calls.append((run_spec, progress, public_compilation_metadata))
        return _application_result()

    fake.run_v04_plane_translation = run_plane_beris_edwards
    monkeypatch.setitem(
        sys.modules,
        "pssolver.runtime.legacy_entry",
        fake,
    )

    result = run_simulation(simulation)

    assert result.source_simulation_sha256 == simulation.canonical_sha256()
    assert len(calls) == 1
    assert calls[0][0].runtime_path.value == "compiled_v2"
    assert calls[0][1] is None
    assert calls[0][2]["normalization"]


def test_public_runner_rejects_incomplete_application_result(
    monkeypatch,
    tmp_path,
):
    compiled = compile_simulation(_plane_simulation(tmp_path))
    fake = ModuleType("pssolver.runtime.legacy_entry")
    fake.run_v04_plane_translation = lambda run_spec, **_kwargs: object()
    monkeypatch.setitem(
        sys.modules,
        "pssolver.runtime.legacy_entry",
        fake,
    )

    with pytest.raises(TypeError, match="missing=.*start_step"):
        run_simulation(compiled)
