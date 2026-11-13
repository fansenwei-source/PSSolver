"""Private reference features are rejected by public construction and restart."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import pssolver as pub
from pssolver.api.lowering import lower_simulation
from pssolver.engine import engine as engine_module
from pssolver.io import engine_state
from pssolver.io.checkpoint import CheckpointCompatibilityError, seal_checkpoint_metadata
from pssolver.planning.assemble import assemble
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05


ROOT = Path(__file__).resolve().parents[1]
BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"
FAMILIES = ("periodic", "plane", "channel")
REJECTION = "engine checkpoint requires the public Euler integrator with empty history"


def test_frozen_blobs():
    paths = ["pssolver/functional/", "pssolver/models/active_nematics/beris_edwards.py",
             "pssolver/models/active_nematics/q_tensor.py"]
    argv = ["git", "ls-tree", "-r", BASE, "--", *paths]
    assert subprocess.check_output(argv, cwd=ROOT) == subprocess.check_output(
        [*argv[:3], "HEAD", *argv[4:]], cwd=ROOT,
    )


def _scene(family, path, **output):
    scene = next(row for row in scenarios() if row["family"] == family
                 and row["dtype"] == "float64" and not row["options"])
    simulation = simulation_v05(scene, path)
    return replace(simulation, output=replace(simulation.output, **output))


@pytest.mark.parametrize("tier", ["modal_block_test", "matrix_free_test", "sbdf2_test"])
def test_private_features_fail_closed(tmp_path, monkeypatch, tier):
    simulation = _scene("periodic", tmp_path)
    problem, _ = lower_simulation(simulation)
    basis = assemble(problem).plan
    if tier == "sbdf2_test":
        target = replace(basis, integrator=replace(basis.integrator, scheme=tier, history_length=1))
    else:
        target = replace(basis, linear=tuple(replace(term, tier=tier) for term in basis.linear))

    def fail(*args, **kwargs):
        pytest.fail("rejected reference feature reached transform construction")

    monkeypatch.setattr(engine_module, "TensorProductTransformBackend", fail)
    with pytest.raises(ValueError, match="^engine supports only registered Periodic Euler plans$"):
        engine_module.Engine(target, device="cpu", dtype=torch.float64)
    with pytest.raises(TypeError, match="^expected a 0.5 Simulation;"):
        pub.run_simulation(target)
    for dry in (False, True):
        declaration = replace(simulation, output=replace(simulation.output, dry_run=dry))
        with pytest.raises(TypeError, match="^unsupported Simulation model declaration$"):
            replace(declaration, model=target)
        with pytest.raises(TypeError, match="^unsupported Simulation time declaration$"):
            replace(declaration, time=target)


def _header(tmp_path, simulation, mutation):
    compiled = pub.compile_simulation(simulation)
    path = tmp_path / "checkpoint"
    path.mkdir()
    plan = compiled.plan.to_metadata()
    history = mutation != "scheme"
    if mutation != "history":
        plan["integrator"]["scheme"] = "sbdf2_test"
    files = {
        key: {"file": f"{key}.npy", "sha256": "0" * 64, "shape": [], "dtype": "float64"}
        for key in ("spectral", "spatial", *(("history_0",) if history else ()))
    }
    metadata = seal_checkpoint_metadata({
        "format_kind": "engine_state", "format_version": 1,
        "plan_identity_sha256": compiled.plan_identity_sha256, "plan": plan,
        "dtype": "float64", "device_kind": "cpu", "batch": 1,
        "completed_steps": 1, "refresh_interval": compiled.plan.integrator.refresh_interval,
        "refresh_phase": [1, 0], "constrained_current": True,
        "history_length": int(history), "solver_keys": [], "tensor_files": files,
        "run_provenance": {},
    })
    (path / "checkpoint.json").write_text(json.dumps(metadata))
    return path


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("dry", [False, True])
@pytest.mark.parametrize("mutation", ["history", "scheme", "both"])
def test_restart_rejected_before_payload(tmp_path, monkeypatch, family, diagnostics, dry, mutation):
    simulation = _scene(family, tmp_path, diagnostics=diagnostics, dry_run=dry)
    path = _header(tmp_path, simulation, mutation)

    def fail(*args, **kwargs):
        pytest.fail("checkpoint rejection touched a payload or the Engine")

    monkeypatch.setattr(engine_state, "resolve_checkpoint_payload", fail)
    monkeypatch.setattr(engine_state, "_hash_file", fail)
    monkeypatch.setattr(engine_state, "load_checkpoint_numpy", fail)
    monkeypatch.setattr(engine_module.Engine, "__init__", fail)
    simulation = replace(simulation, output=replace(simulation.output, restart_from=path))
    with pytest.raises(CheckpointCompatibilityError) as caught:
        pub.run_simulation(simulation)
    assert str(caught.value) == REJECTION
    assert caught.value.operation == "engine_state_import"
    assert caught.value.code == "checkpoint_compatibility_error"
    assert not simulation.output.directory.exists()


def test_public_surface_in_fresh_process():
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);import pssolver;"
        "assert not {'numpy','torch'} & sys.modules.keys();"
        "assert len(pssolver.__all__) == 41;"
        "assert pssolver.capability_catalog().qualified_combinations;"
        "assert not {'numpy','torch'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT)], check=True)
