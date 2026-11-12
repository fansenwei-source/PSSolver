"""Retain the public runner's live contracts on the new declaration path."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import pssolver as public
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "periodic": "periodic_default_float64_default",
    "plane": "plane_legacy_production_float64_default",
    "channel": "channel_default_float64_default",
}


def _declare(tmp_path, family, **output):
    scene = next(row for row in scenarios() if row["id"] == FAMILIES[family])
    simulation = simulation_v05(scene, tmp_path, steps=2)
    return replace(simulation, output=replace(simulation.output, **output))


@pytest.mark.parametrize("family", FAMILIES)
def test_public_declaration_and_plan_integrity_fail_closed(tmp_path, family):
    simulation = _declare(tmp_path, family, dry_run=True)
    compiled = public.compile_simulation(simulation)
    assert compiled.source is simulation
    assert compiled.validate_integrity() is None
    assert compiled.declaration_sha256 == simulation.canonical_sha256()
    assert compiled.plan_identity_sha256 == compiled.plan.identity_sha256
    for field in ("declaration_sha256", "plan_identity_sha256"):
        invalid = replace(compiled, **{field: "0" * 64})
        with pytest.raises(ValueError, match="identity"):
            invalid.validate_integrity()
    with pytest.raises(TypeError, match="product of compile_simulation"):
        public.run_simulation(replace(compiled))
    result = public.run_simulation(compiled)
    assert not simulation.output.directory.exists()
    assert not result.completed
    assert result.final_observation is None
    assert result.saved_steps == result.checkpoint_steps == result.diagnostics == ()


@pytest.mark.parametrize("family", FAMILIES)
def test_public_checkpoint_does_not_change_final_state_and_can_resume(tmp_path, family):
    original = _declare(tmp_path, family, save_interval=1)
    scheduled = replace(original, output=replace(
        original.output, directory=tmp_path / "checkpointed", checkpoint_interval=1,
    ))
    plain, checked = map(public.run_simulation, (original, scheduled))
    assert plain.completed and checked.completed
    assert plain.start_step == checked.start_step == 0
    assert plain.final_step == checked.final_step == 2
    assert checked.checkpoint_steps == (1, 2)
    assert not plain.checkpoint_steps
    checkpoint = next(
        header.parent for header in scheduled.output.directory.rglob("checkpoint.json")
        if json.loads(header.read_bytes())["completed_steps"] == 1
    )
    resumed = public.run_simulation(replace(original, output=replace(
        original.output, directory=tmp_path / "resumed", restart_from=checkpoint, steps=1,
    )))
    assert resumed.start_step == 1 and resumed.final_step == 2
    for field in ("q", "velocity", "pressure"):
        arrays = [getattr(run.final_observation, field) for run in (plain, checked, resumed)]
        records = [(a.dtype, a.shape, a.tobytes(order="C")) for a in arrays]
        assert records[0] == records[1] == records[2]


def test_root_and_catalog_do_not_import_tensor_dependencies():
    program = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pssolver
assert not {'numpy', 'torch'} & set(sys.modules)
root = Path(sys.argv[1])
table = json.loads((root / 'qualification/s28b_tables.json').read_bytes())
assert set(pssolver.__all__) == set(table['public_api']['new_root']) == set(dir(pssolver))
catalog = pssolver.capability_catalog()
assert catalog.qualified_combinations
assert not {'numpy', 'torch'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", program, str(ROOT)], check=True)
