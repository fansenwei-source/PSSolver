"""P7.7.10 common application runner and public result protocol."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys
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
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
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


def _channel_simulation(
    tmp_path: Path,
    *,
    runtime_path: str = "compiled_channel_v2",
) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
        steps=2,
        save_interval=1,
        diagnostic_interval=1,
        generated_output_directory=tmp_path / "generated",
        snapshot_output_directory=tmp_path / "snapshot",
        device="cpu",
        dtype="float32",
        runtime_path=runtime_path,
    )
    source = compose_channel_active_nematics_simulation(request.components)
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=source.invocation,
    )


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


@pytest.mark.parametrize("runtime_path", ("legacy_channel", "compiled_channel_v2"))
def test_run_simulation_dispatches_channel_once_and_returns_public_result(
    monkeypatch,
    tmp_path,
    runtime_path,
):
    simulation = _channel_simulation(tmp_path, runtime_path=runtime_path)
    compiled = compile_simulation(simulation)
    observed = {}
    fake = ModuleType("pssolver.applications.channel_active_nematics")

    def run_channel_active_nematics(run_spec, *, progress):
        observed.update(run_spec=run_spec, progress=progress)
        return _application_result()

    fake.run_channel_active_nematics = run_channel_active_nematics
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.channel_active_nematics",
        fake,
    )
    progress = (0, 1)

    result = run_simulation(compiled, progress=progress)

    assert isinstance(result, SimulationResult)
    assert result.status is SimulationRunStatus.COMPLETE
    assert result.completed is True
    assert result.application == (
        "channel_legacy_active_force_active_nematics"
    )
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
    simulation = _channel_simulation(tmp_path)
    fake = ModuleType("pssolver.applications.channel_active_nematics")
    calls = []

    def run_channel_active_nematics(run_spec, *, progress):
        calls.append((run_spec, progress))
        return _application_result()

    fake.run_channel_active_nematics = run_channel_active_nematics
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.channel_active_nematics",
        fake,
    )

    result = run_simulation(simulation)

    assert result.source_simulation_sha256 == simulation.canonical_sha256()
    assert len(calls) == 1
    assert calls[0][0].runtime_path.value == "compiled_channel_v2"
    assert calls[0][1] is None


def test_public_runner_rejects_incomplete_application_result(
    monkeypatch,
    tmp_path,
):
    compiled = compile_simulation(_channel_simulation(tmp_path))
    fake = ModuleType("pssolver.applications.channel_active_nematics")
    fake.run_channel_active_nematics = lambda run_spec, *, progress: object()
    monkeypatch.setitem(
        sys.modules,
        "pssolver.applications.channel_active_nematics",
        fake,
    )

    with pytest.raises(TypeError, match="missing=.*start_step"):
        run_simulation(compiled)


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
