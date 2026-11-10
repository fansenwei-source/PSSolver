"""S-17 contracts for the D-0017 R3/R6 implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from pssolver.api import capability_catalog
import pssolver.functional.api as stable_functional
from pssolver.experimental import boundary_pilots


ROOT = Path(__file__).resolve().parents[1]


def _isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_stable_functional_child_import_is_silent_but_facade_access_warns():
    result = _isolated_python(
        """
import importlib
import warnings

with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    stable = importlib.import_module("pssolver.functional.api")
assert not observed, observed

compatibility = importlib.import_module("pssolver.functional")
with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    value = compatibility.FunctionalRuntimeIdentity
assert value is stable.FunctionalRuntimeIdentity
assert len(observed) == 1
assert observed[0].category is DeprecationWarning
assert "pssolver.functional.api" in str(observed[0].message)

with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    assert compatibility.FunctionalRuntimeIdentity is value
assert not observed
"""
    )
    assert result.returncode == 0, result.stderr


def test_functional_facade_preserves_raw_builder_compatibility():
    result = _isolated_python(
        """
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import pssolver.functional as compatibility
import pssolver.functional.api as stable
from pssolver.functional import periodic_activity

assert compatibility.build_functional_runtime is periodic_activity.build_functional_runtime
assert compatibility.build_functional_runtime is not stable.build_functional_runtime
assert compatibility.FunctionalCapabilitySet is stable.FunctionalCapabilitySet
assert compatibility.FunctionalCapabilities is stable.FunctionalCapabilitySet
assert all(hasattr(compatibility, name) for name in compatibility.__all__)
"""
    )
    assert result.returncode == 0, result.stderr


def test_explicit_boundary_pilot_namespace_is_warning_free_and_fail_closed():
    result = _isolated_python(
        """
import importlib
import json
import warnings

with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    pilots = importlib.import_module("pssolver.experimental.boundary_pilots")
assert not observed, observed
print(json.dumps(pilots.boundary_pilot_status(), sort_keys=True))
"""
    )
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status == {
        "families": ["robin", "finite_q", "lifting"],
        "legacy_imports_available_through": "all v0.4 releases",
        "lifting_blocked_by": ["B8", "S2-07"],
        "lifting_execution_open": False,
        "namespace": "pssolver.experimental.boundary_pilots",
        "public_simulation_execution_qualified": False,
        "released_checkpoint_readers_preserved": True,
    }

    robin = next(
        item
        for item in capability_catalog().to_metadata()["boundary_policies"]
        if item["key"] == "robin"
    )
    assert robin["declarable"] is True
    assert robin["executable"] is False
    assert robin["qualified_applications"] == []


def test_old_pilot_facades_warn_and_resolve_to_the_explicit_namespace():
    result = _isolated_python(
        """
import warnings
from pssolver.experimental import boundary_pilots as pilots
import pssolver.operators as operators
import pssolver.planning as planning
import pssolver.runtime as runtime
import pssolver.workflows as workflows

pairs = (
    (planning, "StaticLiftingPlan", pilots.StaticLiftingPlan),
    (operators, "CellCenteredRobinEigenbasisOperator", pilots.CellCenteredRobinEigenbasisOperator),
    (runtime, "PlaneRobinScalarRuntime", pilots.PlaneRobinScalarRuntime),
    (workflows, "PlaneFiniteQAnchoringWorkflow", pilots.PlaneFiniteQAnchoringWorkflow),
)
with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    for module, name, expected in pairs:
        assert getattr(module, name) is expected
assert len(observed) == len(pairs)
assert all(item.category is DeprecationWarning for item in observed)
assert all("pssolver.experimental.boundary_pilots" in str(item.message) for item in observed)
"""
    )
    assert result.returncode == 0, result.stderr


def test_plain_package_import_preserves_v030_submodule_attributes_without_warning():
    result = _isolated_python(
        """
import importlib
import json
import types
import warnings

expected = {
    "pssolver.functional": [
        "channel_activity",
        "channel_activity_runtime",
        "channel_checkpoint",
        "channel_pressure",
        "channel_pressure_adjoint",
        "checkpoint_identity",
        "contracts",
        "errors",
        "legacy_identity",
        "periodic_activity",
        "periodic_checkpoint",
        "pressure_metadata",
        "validation",
        "versioning",
    ],
    "pssolver.operators": [
        "lifting",
        "projection",
        "robin",
        "tensor_divergence",
    ],
    "pssolver.runtime": [
        "channel_beris_edwards",
        "finite_q_anchoring",
        "package_construction",
        "plane_application_bridge",
        "plane_beris_edwards",
        "plane_legacy",
        "robin_scalar",
        "simulation_construction",
        "state_validation",
        "static_lifting",
    ],
    "pssolver.workflows": [
        "channel_beris_edwards",
        "channel_checkpoint_observation",
        "finite_q_anchoring",
        "plane_beris_edwards",
        "plane_checkpoint",
        "plane_observation",
    ],
}
observed = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for package_name, names in expected.items():
        package = importlib.import_module(package_name)
        for name in names:
            module = getattr(package, name)
            assert isinstance(module, types.ModuleType)
            assert module.__name__ == f"{package_name}.{name}"
        observed[package_name] = sorted(
            name
            for name, value in vars(package).items()
            if isinstance(value, types.ModuleType)
            and value.__name__.startswith(f"{package_name}.")
        )
        assert "facade_dir" not in dir(package)
        assert "resolve_deprecated_facade_name" not in dir(package)
assert not caught, [str(item.message) for item in caught]
assert observed == expected
print(json.dumps(observed, sort_keys=True))
"""
    )
    assert result.returncode == 0, result.stderr


def test_released_finite_q_reader_keeps_its_defining_module_identity():
    from pssolver.workflows.finite_q_anchoring import (
        load_finite_q_anchoring_checkpoint,
    )

    assert (
        boundary_pilots.load_finite_q_anchoring_checkpoint
        is load_finite_q_anchoring_checkpoint
    )
    assert load_finite_q_anchoring_checkpoint.__module__ == (
        "pssolver.workflows.finite_q_anchoring"
    )


def test_control_first_functional_and_planar_boundaries_are_unchanged():
    assert stable_functional.FUNCTIONAL_API_VERSION == "1.1"
    expected_blobs = {
        "pssolver/models/active_nematics/beris_edwards.py": (
            "07f1f03f7f2f35e28e8ce629c1d73c5d59e4f5f3"
        ),
        "pssolver/models/active_nematics/q_tensor.py": (
            "5427df4402f31fc43ac409e1936cd31e58269966"
        ),
    }
    for relative, expected in expected_blobs.items():
        result = subprocess.run(
            ["git", "hash-object", relative],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == expected
