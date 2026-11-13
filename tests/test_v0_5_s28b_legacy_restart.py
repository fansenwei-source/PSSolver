"""Pin v0.4 declaration bytes and the first resumed Plane diagnostic."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pssolver as public
from pssolver.api.legacy_checkpoints import downgrade_simulation
from pssolver.cli.plane import _parser, _simulation, main
from pssolver.core._canonical import canonical_json
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05


FIXTURES = Path(__file__).with_name("fixtures")
REFERENCE = json.loads((FIXTURES / "s28b_downgrade_v04_canonical.json").read_bytes())
CASES = {row["id"]: row for row in scenarios()}
PLANE_FIXTURE = FIXTURES / "v0_4_0_s23" / "plane_legacy_production_float64_default"
PLANE_OPTIONS = (
    "--activity-number", "18", "--parameterization", "fixed-k", "--frank-k", repr(1 / 81),
    "--lx", "16", "--ly", "16", "--height", "12", "--nx", "6", "--ny", "6", "--nz", "4",
    "--dt", "0.001", "--num-defect-pairs", "1", "--save-start-step", "0",
    "--dtype", "float64", "--device", "cpu", "--pointwise-execution", "eager", "--tf32", "off",
    "--diagnostics", "--diagnostic-interval", "1", "--checkpoint-interval", "1", "--save-interval", "1",
)


def test_downgrade_matches_the_installed_v04_canonical_matrix(tmp_path):
    rows = REFERENCE["rows"]
    authority = json.loads((FIXTURES.parents[1] / "qualification/s28b_tables.json").read_bytes())
    production = {name for name in authority["plan_identity_reference"] if not name.startswith("functional_")}
    assert set(rows) == production
    assert len(rows) == REFERENCE["scenes"] == 98
    assert REFERENCE["work_placeholder"] == "<WORK>"
    for name, row in rows.items():
        work = tmp_path / name
        work.mkdir()
        scene = CASES[name]
        old = downgrade_simulation(simulation_v05(scene, work), runtime_path=scene["runtime_path"])
        metadata = old.to_metadata()
        payload = canonical_json(metadata)
        assert payload.count(str(tmp_path)) == row["work_occurrences"], name
        reference = row["canonical_normalized"]
        assert hashlib.sha256(reference.encode()).hexdigest() == row["sha256_normalized"], name
        normalized = json.loads(payload.replace(str(tmp_path), "<WORK>"))
        expected = json.loads(reference)
        if scene["family"] != "plane":
            # The archived run digest still includes its original absolute paths.
            # Check the candidate's raw digest, then derive both from normalized bytes.
            raw_run = metadata["identities"]["run"]
            assert raw_run["sha256"] == _canonical_digest(raw_run["metadata"]), name
            for value in (normalized, expected):
                run = value["identities"]["run"]
                run["sha256"] = _canonical_digest(run["metadata"])
        assert canonical_json(normalized).encode() == canonical_json(expected).encode(), name


def _canonical_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@pytest.mark.parametrize("family", ["periodic", "channel"])
def test_downgrade_preserves_true_dry_run_digest_and_checkpoint_path(tmp_path, family):
    scene = CASES[f"{family}_default_float64_default"]
    source = simulation_v05(scene, tmp_path)
    default = downgrade_simulation(source)
    assert dict(default.invocation.options) == {}
    digest = "a" * 64
    checkpoint = tmp_path / "checkpoint_1"
    explicit = replace(source, output=replace(
        source.output, dry_run=True, validation_config_sha256=digest, restart_from=checkpoint,
    ))
    legacy = downgrade_simulation(explicit).to_metadata()
    assert legacy["invocation"] == {"dry_run": True, "validation_config_sha256": digest}
    assert legacy["workflow"]["options"]["restart_from"] == str(checkpoint)


def _plane_args(output, *extra):
    return [*PLANE_OPTIONS, "--steps", "1", "--output-dir", str(output), *map(str, extra)]


def _plane(output, *extra):
    return _simulation(_parser().parse_args(_plane_args(output, *extra)))


def _diagnostics(output):
    with (output / "diagnostics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_plane_engine_state_restart_publishes_the_restored_diagnostic(tmp_path):
    fresh = _plane(tmp_path / "fresh")
    first = public.run_simulation(fresh)
    assert first.checkpoint_steps == (1,)
    resumed = _plane(tmp_path / "resumed", "--restart-from", fresh.output.directory / "checkpoint_1")
    result = public.run_simulation(resumed)
    assert (result.start_step, result.final_step) == (1, 2)
    assert [row.step for row in result.diagnostics] == [1, 2]
    assert _diagnostics(resumed.output.directory)[0] == _diagnostics(fresh.output.directory)[-1]


def test_plane_legacy_restart_publishes_the_restored_diagnostic(tmp_path):
    resumed = _plane(tmp_path / "resumed", "--restart-from", PLANE_FIXTURE)
    result = public.run_simulation(resumed)
    assert result.completed
    assert [row.step for row in result.diagnostics] == [result.start_step, result.final_step]
    assert len(_diagnostics(resumed.output.directory)) == 2


def test_plane_cli_restart_publishes_the_restored_diagnostic(tmp_path):
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    assert main(_plane_args(first)) == 0
    assert main(_plane_args(resumed, "--restart-from", first / "checkpoint_1")) == 0
    assert _diagnostics(resumed)[0] == _diagnostics(first)[-1]
