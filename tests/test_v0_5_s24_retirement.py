"""S-24 retirement boundaries; retained physics stays in the S-23 replay gates."""

import ast
import hashlib
import inspect
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pytest

import pssolver
from pssolver.functional.legacy_declarations.legacy_plane import read_plane_checkpoint_header
from pssolver.io import CheckpointCompatibilityError
from pssolver.linear_solvers.stokes import _pressure_policy


ROOT = Path(__file__).resolve().parents[1]
STAGES = json.loads((ROOT / "qualification/stages/manifest.json").read_text())
RELEASE = "4c12228aba01e8483e80d28a1471ab72384d5fda"
A_NAMES = {
    "stage_m_plan",
    "stage_n_plan",
    "stage_n1_plan",
    "stage_n2_plan",
    "stage_n3_plan",
    "stage_n4_plan",
    "stage_o4_plan",
}


def test_public_catalog_contains_only_the_three_retained_combinations():
    catalog = pssolver.capability_catalog()
    assert {
        (row.equation_variant, row.geometry_name, row.boundary_policies)
        for row in catalog.qualified_combinations
    } == {
        ("complete_stress_beris_edwards", "plane_slab", (
            "neumann_q", "free_slip_velocity", "neumann_pressure_compatibility",
        )),
        ("complete_stress_beris_edwards", "periodic_box", (
            "neumann_q", "free_slip_velocity", "neumann_pressure_compatibility",
        )),
        ("complete_stress_beris_edwards", "periodic_box", (
            "neumann_q", "no_slip_velocity", "neumann_pressure_compatibility",
        )),
        ("complete_stress_beris_edwards", "rectangular_channel", (
            "neumann_q", "no_slip_velocity", "neumann_pressure_compatibility",
        )),
        ("shendruk_plane_beris_edwards", "plane_slab", (
            "neumann_q", "free_slip_velocity", "neumann_pressure_compatibility",
        )),
    }
    assert all(row.qualified for row in catalog.qualified_combinations)


def test_missing_historical_git_is_an_explicit_prerequisite_failure(tmp_path, monkeypatch):
    from qualification import historical_stage_audit

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(historical_stage_audit.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="PSSOLVER_TEST_PREREQUISITE"):
        historical_stage_audit._git(tmp_path, "rev-parse", "HEAD")


def test_stage_move_is_byte_identical_and_allowlist_is_exact():
    from qualification._legacy_imports import ALLOWED

    assert ALLOWED == A_NAMES
    assert len(STAGES) == 35
    assert sum(item["group"] == "A" for item in STAGES) == 7
    assert sum(item["group"] == "B" for item in STAGES) == 28
    for item in STAGES:
        assert (
            hashlib.sha256((ROOT / item["destination"]).read_bytes()).hexdigest() == item["sha256"]
        )
        assert not (ROOT / item["source"]).exists()
        assert (Path(item["source"]).stem in ALLOWED) == (item["group"] == "A")


@pytest.mark.parametrize("item", STAGES, ids=lambda item: Path(item["source"]).stem)
def test_fresh_interpreter_stage_import_boundary(item):
    source = """
import importlib
import importlib.util
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
expected_provider = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(expected_provider.parent.parent))
import pssolver
assert Path(pssolver.__file__).resolve() == expected_provider
sys.path.insert(0, str(root))
from qualification._legacy_imports import install
install()
manifest = json.loads((root / 'qualification/stages/manifest.json').read_text())
name, group = sys.argv[2:4]
module = 'pssolver.experimental.' + name
if group == 'A':
    importlib.import_module(module)
else:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        assert error.name == module, (module, error.name)
    else:
        raise AssertionError('B-group unexpectedly imported')
assert not { 'pssolver.experimental.' + Path(item['source']).stem
             for item in manifest if item['group'] == 'B' }.intersection(sys.modules)
for retired in json.loads((root / 'qualification/s24_retired_modules.json').read_text()):
    assert retired not in sys.modules, retired
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            source,
            str(ROOT),
            Path(item["source"]).stem,
            item["group"],
            pssolver.__file__,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_never_imports_stage_artifacts_or_retired_modules():
    retired = set(json.loads((ROOT / "qualification/s24_retired_modules.json").read_text()))
    retired.update("pssolver.experimental." + Path(item["source"]).stem for item in STAGES)
    for path in (ROOT / "pssolver").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = {name.name for name in node.names}
            elif isinstance(node, ast.ImportFrom):
                import importlib.util

                target = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                targets = {target, *(target + "." + name.name for name in node.names)}
            else:
                continue
            assert not targets.intersection(retired), (path, node.lineno, targets & retired)


def _released_policy():
    source = subprocess.check_output(
        [
            "git",
            "show",
            RELEASE + ":pssolver/configuration/channel_active_nematics_declarations.py",
        ],
        cwd=ROOT,
        text=True,
    )
    names = {
        "CHANNEL_PRESSURE_POLICY_SCHEMA_VERSION",
        "CHANNEL_PRESSURE_RHS_NORM_DEFINITION",
        "CHANNEL_PRESSURE_WORKING_VECTOR_SCALING",
        "CHANNEL_PRESSURE_BREAKDOWN_DEFINITION",
        "_CHANNEL_PRESSURE_INITIAL_GUESS_POLICIES",
        "_finite",
        "_positive",
        "_integer",
        "channel_pressure_policy_metadata",
    }
    nodes = [
        node
        for node in ast.parse(source).body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    assert len(nodes) == len(names)
    namespace = {"math": math, "Integral": Integral, "Real": Real}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<v0.4.0-policy>", "exec"), namespace)
    for name in ("_finite", "_positive", "_integer", "channel_pressure_policy_metadata"):
        original = next(node for node in nodes if getattr(node, "name", None) == name)
        current = ast.parse(inspect.getsource(getattr(_pressure_policy, name))).body[0]
        assert ast.dump(original) == ast.dump(current)
    return namespace["channel_pressure_policy_metadata"]


def _outcome(call, values):
    try:
        value = call(**values)
    except Exception as error:
        return (type(error).__module__, type(error).__name__, str(error))
    return json.dumps(value, separators=(",", ":"))


def test_pressure_policy_output_and_validation_match_released_source_exactly():
    released = _released_policy()
    baseline = dict(
        relative_tolerance=1e-8,
        max_iterations=200,
        fixed_iterations=None,
        initial_guess_policy="persistent_production_warm_start",
    )
    matrix = [baseline]
    for key, values in {
        "relative_tolerance": [0, -1, True, "1", None, math.nan, math.inf, 1e-6],
        "max_iterations": [0, -1, True, 1.5, None, 300],
        "fixed_iterations": [0, -1, True, 1.5, None, 200],
        "initial_guess_policy": ["unknown", None, "zero_every_call"],
    }.items():
        matrix.extend({**baseline, key: value} for value in values)
    for values in matrix:
        assert _outcome(_pressure_policy.channel_pressure_policy_metadata, values) == _outcome(
            released, values
        )


@pytest.mark.parametrize(
    "relative,family",
    [
        ("v0_2_0_checkpoints_v1/production_legacy_channel", "legacy active-force Channel"),
        ("v0_4_0_s24_retired/separated_canary", "separated_canary"),
    ],
)
def test_retired_checkpoint_rejected_before_any_tensor_access(relative, family, monkeypatch):
    directory = ROOT / "tests/fixtures" / relative
    real_open = Path.open
    opened = []

    def metadata_only(path, *args, **kwargs):
        opened.append(path)
        assert path.name == "checkpoint.json", f"payload accessed before rejection: {path}"
        return real_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("tensor loader was reached before retired-family rejection")

    monkeypatch.setattr(Path, "open", metadata_only)
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(CheckpointCompatibilityError, match=family) as caught:
        read_plane_checkpoint_header(directory)
    assert "v0.4.x (maint/v0.4)" in str(caught.value)
    assert opened == [directory / "checkpoint.json"]


def test_current_compiled_rhs_metadata_still_matches_frozen_q4_contract():
    from pssolver.models.active_nematics.beris_edwards import BerisEdwardsPointwiseKernels

    source = (ROOT / "qualification/stages/stage_q4_qualification.py").read_text()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_compiled_rhs"
    )
    namespace = {"Mapping": Mapping}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<frozen-q4-validator>", "exec"),
        namespace,
    )
    validate = namespace["_validate_compiled_rhs"]
    metadata = {
        "owner": "geometry_executor",
        "implementation_name": "plane_beris_edwards_pointwise_explicit_rhs",
        "observability": {
            "fallback_to_model": False,
            "pointwise_kernels": BerisEdwardsPointwiseKernels("compile").metadata(),
        },
    }
    assert validate(metadata)
    pointwise = metadata["observability"]["pointwise_kernels"]
    pointwise["compile"]["fallback_allowed"] = pointwise.pop("fallback_allowed")
    pointwise["compile"]["fallback_reason"] = pointwise.pop("fallback_reason")
    assert not validate(metadata)
