"""RC4.2.1 typed identity primitives and frozen legacy registry."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import pssolver.io as public_io
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    CheckpointIdentityLayer,
    IdentityLayerKind,
    LEGACY_IDENTITY_SCHEMAS,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    legacy_identity_registry,
    legacy_registry_metadata,
    legacy_registry_sha256,
    resolve_legacy_identity_schema,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "pssolver" / "io" / "checkpoint_identity.py"
ORACLE = (
    ROOT
    / "tests"
    / "fixtures"
    / "checkpoint_identity"
    / "legacy_registry_v1.json"
)


def _oracle() -> dict[str, object]:
    return json.loads(ORACLE.read_text(encoding="utf-8"))


def _layer(kind, version, payload):
    return CheckpointIdentityLayer(
        kind=kind,
        version=version,
        payload=payload,
    )


def _plane_identity():
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PLANE_WORKFLOW,
        runtime_path="legacy_production",
        forward_dynamics=_layer(
            IdentityLayerKind.FORWARD_DYNAMICS,
            "plane.forward.v1",
            {
                "dt": 0.01,
                "device_excluded": True,
                "semantics": ["q", "stokes"],
            },
        ),
        state_layout=_layer(
            IdentityLayerKind.STATE_LAYOUT,
            "plane.state.v1",
            {
                "components": ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"],
                "representation": "spatial_and_spectral",
            },
        ),
        backend_restart=_layer(
            IdentityLayerKind.BACKEND_RESTART,
            "plane.backend.v1",
            {"state_keys": []},
        ),
    )


def _channel_functional_identity():
    return CheckpointCompatibilityIdentity(
        family=CheckpointFamily.CHANNEL_FUNCTIONAL,
        runtime_path="channel_activity_batch_one",
        forward_dynamics=_layer(
            IdentityLayerKind.FORWARD_DYNAMICS,
            "channel.forward.v2",
            {
                "forward_stokes_nyquist": (
                    "project_even_periodic_plane"
                )
            },
        ),
        derivative_dynamics=_layer(
            IdentityLayerKind.DERIVATIVE_DYNAMICS,
            "channel.derivative.v2",
            {
                "pressure_transpose_nyquist": (
                    "project_even_periodic_plane"
                ),
                "gradient": "custom_implicit_pressure_adjoint_v1",
            },
        ),
        state_layout=_layer(
            IdentityLayerKind.STATE_LAYOUT,
            "channel.functional.state.v1",
            {"components": ["q_physical", "q_spectral"]},
        ),
        backend_restart=_layer(
            IdentityLayerKind.BACKEND_RESTART,
            "channel.functional.backend.v1",
            {"state_keys": []},
        ),
    )


def test_identity_layers_are_canonical_deeply_immutable_and_typed():
    source = {
        "nested": {"b": 2, "a": 1},
        "components": ["Qxx", "Qxy"],
    }
    first = _layer(
        IdentityLayerKind.FORWARD_DYNAMICS,
        "plane.forward.v1",
        source,
    )
    second = _layer(
        IdentityLayerKind.FORWARD_DYNAMICS,
        "plane.forward.v1",
        {
            "components": ["Qxx", "Qxy"],
            "nested": {"a": 1, "b": 2},
        },
    )

    source["nested"]["a"] = 99
    source["components"].append("Qxz")
    assert first.to_metadata() == second.to_metadata()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert hash(first) == hash(second)
    assert first.payload["nested"]["a"] == 1
    assert first.payload["components"] == ("Qxx", "Qxy")
    with pytest.raises(TypeError):
        first.payload["nested"]["a"] = 3
    assert first.checkpoint_compatibility_gate is True
    provenance = _layer(
        IdentityLayerKind.RUN_PROVENANCE,
        "run.v1",
        {"requested_device": "cuda:0"},
    )
    assert provenance.checkpoint_compatibility_gate is False


def test_compatibility_identity_excludes_provenance_by_layer_kind():
    plane = _plane_identity()
    forward = plane.forward_dynamics
    provenance = _layer(
        IdentityLayerKind.MATERIALIZATION_PROVENANCE,
        "materialization.v1",
        {"device": "cuda:0"},
    )

    with pytest.raises(ValueError, match="expected forward_dynamics"):
        CheckpointCompatibilityIdentity(
            family=CheckpointFamily.PLANE_WORKFLOW,
            runtime_path="legacy_production",
            forward_dynamics=provenance,
            state_layout=plane.state_layout,
            backend_restart=plane.backend_restart,
        )
    rebuilt = CheckpointCompatibilityIdentity(
        family=CheckpointFamily.PLANE_WORKFLOW,
        runtime_path="legacy_production",
        forward_dynamics=forward,
        state_layout=plane.state_layout,
        backend_restart=plane.backend_restart,
    )
    assert rebuilt.canonical_sha256() == plane.canonical_sha256()
    assert hash(rebuilt) == hash(plane)


def test_identity_metadata_round_trips_through_strict_readers():
    layer = _plane_identity().forward_dynamics
    identity = _plane_identity()

    assert CheckpointIdentityLayer.from_metadata(layer.to_metadata()) == layer
    assert CheckpointCompatibilityIdentity.from_metadata(
        identity.to_metadata()
    ) == identity


@pytest.mark.parametrize("kind", ("layer", "aggregate"))
def test_identity_metadata_readers_reject_unknown_keys(kind):
    value = (
        _plane_identity().forward_dynamics.to_metadata()
        if kind == "layer"
        else _plane_identity().to_metadata()
    )
    value["unknown"] = True
    reader = (
        CheckpointIdentityLayer.from_metadata
        if kind == "layer"
        else CheckpointCompatibilityIdentity.from_metadata
    )

    with pytest.raises(ValueError, match="metadata keys"):
        reader(value)


def test_functional_identity_requires_derivative_dynamics_and_production_forbids_it():
    plane = _plane_identity()
    derivative = _layer(
        IdentityLayerKind.DERIVATIVE_DYNAMICS,
        "test.derivative.v1",
        {"gradient": "test"},
    )

    with pytest.raises(ValueError, match="must not carry derivative"):
        CheckpointCompatibilityIdentity(
            family=CheckpointFamily.PLANE_WORKFLOW,
            runtime_path="legacy_production",
            forward_dynamics=plane.forward_dynamics,
            derivative_dynamics=derivative,
            state_layout=plane.state_layout,
            backend_restart=plane.backend_restart,
        )
    with pytest.raises(ValueError, match="require derivative"):
        CheckpointCompatibilityIdentity(
            family=CheckpointFamily.CHANNEL_FUNCTIONAL,
            runtime_path="channel_activity_batch_one",
            forward_dynamics=plane.forward_dynamics,
            state_layout=plane.state_layout,
            backend_restart=plane.backend_restart,
        )
    assert _channel_functional_identity().derivative_dynamics is not None


@pytest.mark.parametrize(
    ("version", "payload", "error"),
    [
        ("BadVersion", {}, ValueError),
        ("valid.v1", {1: "non-string key"}, TypeError),
        ("valid.v1", {"value": float("nan")}, ValueError),
        ("valid.v1", {"value": object()}, TypeError),
    ],
)
def test_identity_layers_reject_noncanonical_values(version, payload, error):
    with pytest.raises(error):
        _layer(IdentityLayerKind.STATE_LAYOUT, version, payload)


def test_identity_oracles_are_frozen_independently_of_live_run_specs():
    expected = {item["name"]: item for item in _oracle()["identity_oracles"]}
    actual = {
        "plane_production_v1": _plane_identity(),
        "channel_functional_v2": _channel_functional_identity(),
    }

    assert set(actual) == set(expected)
    for name, identity in actual.items():
        assert identity.to_metadata() == expected[name]["metadata"]
        assert identity.canonical_sha256() == expected[name]["sha256"]


def test_legacy_registry_has_exact_releases_families_and_runtime_paths():
    oracle = _oracle()["registry"]
    registry = legacy_identity_registry()
    counts = Counter(entry.key.checkpoint_family.value for entry in registry.values())

    assert len(registry) == oracle["entry_count"] == 21
    assert counts == oracle["family_counts"]
    assert legacy_registry_metadata()["entry_count"] == len(registry)
    assert legacy_registry_sha256() == oracle["sha256"]
    assert tuple(registry.values()) == LEGACY_IDENTITY_SCHEMAS
    assert {
        entry.key.source_release_generation.value: entry.source_commit
        for entry in LEGACY_IDENTITY_SCHEMAS
    } == oracle["release_commits"]
    plane_paths = {
        entry.key.runtime_path
        for entry in LEGACY_IDENTITY_SCHEMAS
        if entry.key.checkpoint_family is CheckpointFamily.PLANE_WORKFLOW
    }
    assert plane_paths == {
        "legacy_production",
        "compiled_v2",
        "separated_canary",
    }
    with pytest.raises(TypeError):
        registry[next(iter(registry))] = next(iter(registry.values()))


def test_registry_source_hashes_are_the_released_files_not_live_derivations():
    source_path = {
        CheckpointFamily.PLANE_WORKFLOW: (
            "pssolver/configuration/plane_beris_edwards_schema_v1.py"
        ),
        CheckpointFamily.PERIODIC_WORKFLOW: (
            "pssolver/configuration/periodic_beris_edwards.py"
        ),
        CheckpointFamily.CHANNEL_WORKFLOW: (
            "pssolver/configuration/channel_beris_edwards.py"
        ),
        CheckpointFamily.PERIODIC_FUNCTIONAL: (
            "pssolver/functional/versioning.py"
        ),
        CheckpointFamily.CHANNEL_FUNCTIONAL: (
            "pssolver/functional/versioning.py"
        ),
    }
    observed: dict[tuple[str, str], str] = {}
    for entry in LEGACY_IDENTITY_SCHEMAS:
        path = source_path[entry.key.checkpoint_family]
        lookup = (entry.source_commit, path)
        if lookup not in observed:
            payload = subprocess.run(
                ["git", "show", f"{entry.source_commit}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            observed[lookup] = hashlib.sha256(payload).hexdigest()
        assert observed[lookup] == entry.source_schema_sha256


def test_registry_resolution_is_exact_and_unknown_keys_fail_closed():
    known = LEGACY_IDENTITY_SCHEMAS[0]
    assert resolve_legacy_identity_schema(known.key) is known
    unknown = LegacyIdentityKey(
        checkpoint_family=CheckpointFamily.PLANE_WORKFLOW,
        format_version=99,
        source_release_generation=SourceReleaseGeneration.RC1,
        runtime_path="legacy_production",
        applicability_class="configuration_dependent",
    )
    with pytest.raises(KeyError, match="unregistered legacy"):
        resolve_legacy_identity_schema(unknown)
    with pytest.raises(TypeError, match="LegacyIdentityKey"):
        resolve_legacy_identity_schema(known.key.to_metadata())


def test_identity_primitives_are_schema_independent_and_production_is_wired():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    assert not any(name.startswith("pssolver") for name in imports)

    wired_readers = [
        "pssolver/workflows/plane_checkpoint.py",
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/workflows/channel_beris_edwards.py",
    ]
    for relative in wired_readers:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "pssolver.io.checkpoint_identity" in source

    untouched_readers = [
        "pssolver/workflows/channel_checkpoint.py",
        "pssolver/functional/periodic_checkpoint.py",
        "pssolver/functional/channel_checkpoint.py",
    ]
    for relative in untouched_readers:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "checkpoint_identity" not in source
    assert "CheckpointCompatibilityIdentity" not in public_io.__all__
    assert "legacy_identity_registry" not in public_io.__all__
