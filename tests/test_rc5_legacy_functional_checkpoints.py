"""R5-1 regression tests against checkpoints written by rc1, rc2, and rc3."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.functional import (
    FunctionalCheckpointCompatibilityError,
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_request,
    periodic_activity_functional_request,
)
from pssolver.functional.legacy_identity import (
    legacy_runtime_identity_sha256,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards, Q_COMPONENTS


_FIXTURE_ROOT = (
    Path(__file__).parent
    / "fixtures"
    / "functional_checkpoint_releases_v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tree_sha256(directory: Path) -> str:
    encoded = json.dumps(
        _tree(directory),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.05,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )


def _output(directory: Path) -> Output:
    return Output(
        directory=directory,
        steps=1,
        save_interval=100,
        diagnostic_interval=100,
        save_start_step=0,
        diagnostics=True,
        save_hydrodynamics=True,
    )


def _periodic_runtime(tmp_path: Path):
    shape = (6, 6, 4)
    model = _model()
    geometry = PeriodicBox(shape=shape, lengths=(6.0, 6.0, 4.0))
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.01, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            _FIXTURE_ROOT / "snapshots" / "snap_periodic",
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=_output(tmp_path / "periodic_output"),
    )
    request = periodic_activity_functional_request(simulation.specification)
    return build_functional_runtime(request)


def _channel_runtime(tmp_path: Path):
    shape = (6, 6, 6)
    model = _model()
    geometry = RectangularChannel(shape=shape, lengths=(6.0, 6.0, 6.0))
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": no_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.01),
        initial_condition=SnapshotInitialCondition(
            _FIXTURE_ROOT / "snapshots" / "snap_channel_6",
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=_output(tmp_path / "channel_output"),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
        invocation=InvocationSpec({}),
    )
    request = channel_activity_functional_request(simulation.specification)
    return build_channel_activity_functional_runtime(request)


@pytest.fixture(scope="module")
def release_runtimes(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy_functional_runtime")
    return _periodic_runtime(root), _channel_runtime(root)


def _periodic_fixture_state(directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.stack(
        tuple(
            torch.from_numpy(np.load(directory / f"spatial__{name}.npy"))
            for name in Q_COMPONENTS
        ),
        dim=0,
    )
    spectral = torch.stack(
        tuple(
            torch.from_numpy(np.load(directory / f"spectral__{name}.npy"))
            for name in Q_COMPONENTS
        ),
        dim=0,
    )
    return spatial, spectral


def _channel_fixture_state(directory: Path) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.load(directory / "state__q_physical.npy")),
        torch.from_numpy(np.load(directory / "state__q_spectral.npy")),
    )


def _assert_state_equal(left, right) -> None:
    assert all(
        torch.equal(a, b)
        for a, b in zip(left, right, strict=True)
    )


def test_release_checkpoint_fixture_is_provenance_bound():
    manifest = json.loads((_FIXTURE_ROOT / "manifest.json").read_text())
    assert manifest["fixture_schema_version"] == 1
    for generation in ("rc1", "rc2", "rc3"):
        info = _FIXTURE_ROOT / generation / "info.json"
        assert _sha256(info) == manifest["releases"][generation][
            "info_sha256"
        ]
        release = json.loads(info.read_text())
        assert release["version"] == f"0.2.0{generation}"
        for family, dirname in (
            ("periodic", "periodic_func_ck"),
            ("channel", "channel_func_ck"),
        ):
            record = manifest["releases"][generation][family]
            directory = _FIXTURE_ROOT / generation / dirname
            assert len(_tree(directory)) == record["file_count"]
            assert _tree_sha256(directory) == record["tree_sha256"]
            assert _sha256(directory / "checkpoint.json") == record[
                "checkpoint_sha256"
            ]
            bridge = json.loads(
                (directory / "checkpoint.json").read_text()
            )["functional_bridge"]
            assert bridge["api_version"] == "1.0"
            assert bridge["functional_runtime_identity_sha256"] == record[
                "functional_runtime_identity_sha256"
            ]


@pytest.mark.parametrize("generation", ("rc2", "rc3"))
def test_real_periodic_v2_checkpoint_is_directly_readable_and_exactly_upgradable(
    release_runtimes,
    tmp_path,
    generation,
):
    periodic, _ = release_runtimes
    source = _FIXTURE_ROOT / generation / "periodic_func_ck"
    before = _tree(source)
    expected = _periodic_fixture_state(source)

    direct = periodic.checkpoint_bridge.import_checkpoint(source)
    assert direct.completed_steps == 2
    assert direct.source_format == "periodic_functional_bridge_v2"
    _assert_state_equal(direct.state, expected)

    target = periodic.checkpoint_bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / f"periodic_{generation}_v3",
        source_release_generation=generation,
    )
    upgraded = periodic.checkpoint_bridge.import_checkpoint(target)
    _assert_state_equal(upgraded.state, expected)
    metadata = json.loads((target / "checkpoint.json").read_text())
    assert metadata["migration_provenance"]["migration_kind"] == (
        "exact_legacy_upgrade"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is True
    assert _tree(source) == before


def test_real_periodic_rc1_checkpoint_uses_explicit_state_only_migration(
    release_runtimes,
    tmp_path,
):
    periodic, _ = release_runtimes
    source = _FIXTURE_ROOT / "rc1" / "periodic_func_ck"
    before = _tree(source)
    expected = _periodic_fixture_state(source)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="explicit state migration",
    ):
        periodic.checkpoint_bridge.import_checkpoint(source)

    target = periodic.checkpoint_bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / "periodic_rc1_v3",
        source_release_generation="rc1",
    )
    restored = periodic.checkpoint_bridge.import_checkpoint(target)
    assert torch.equal(restored.state[0], expected[0])
    assert restored.compatibility.exact_current_protocol is False
    metadata = json.loads((target / "checkpoint.json").read_text())
    assert metadata["migration_provenance"]["migration_kind"] == (
        "state_only_migration"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is False
    assert _tree(source) == before


def test_stripped_periodic_rc1_functional_checkpoint_cannot_downgrade(
    release_runtimes,
    tmp_path,
    monkeypatch,
):
    periodic, _ = release_runtimes
    source = _FIXTURE_ROOT / "rc1" / "periodic_func_ck"
    stripped = tmp_path / "periodic_rc1_stripped"
    shutil.copytree(source, stripped)
    metadata_path = stripped / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("functional_bridge")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = stripped / "functional_checkpoint.json"
    if marker.exists():
        marker.unlink()
    monkeypatch.setattr(
        "pssolver.workflows.periodic_checkpoint._load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="production-v1 checkpoints are not accepted",
    ) as error:
        periodic.checkpoint_bridge.import_checkpoint(stripped)
    assert error.value.code == "functional_checkpoint_compatibility_error"


@pytest.mark.parametrize(
    ("generation", "migration_kind", "trajectory_equivalent"),
    (
        ("rc1", "state_only_migration", False),
        ("rc2", "state_only_migration", False),
        ("rc3", "exact_legacy_upgrade", True),
    ),
)
def test_real_channel_checkpoint_obeys_release_generation_migration_contract(
    release_runtimes,
    tmp_path,
    generation,
    migration_kind,
    trajectory_equivalent,
):
    _, channel = release_runtimes
    source = _FIXTURE_ROOT / generation / "channel_func_ck"
    before = _tree(source)
    expected = _channel_fixture_state(source)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="requires explicit migration",
    ):
        channel.checkpoint_bridge.import_checkpoint(source)

    target = channel.checkpoint_bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / f"channel_{generation}_v3",
        source_release_generation=generation,
    )
    restored = channel.checkpoint_bridge.import_checkpoint(target)
    _assert_state_equal(restored.state, expected)
    metadata = json.loads((target / "checkpoint.json").read_text())
    assert metadata["migration_provenance"]["migration_kind"] == migration_kind
    assert (
        metadata["migration_provenance"]["trajectory_equivalent"]
        is trajectory_equivalent
    )
    assert _tree(source) == before


@pytest.mark.parametrize(
    ("family", "generation", "manifest_key"),
    (
        ("periodic", "rc1", "periodic"),
        ("periodic", "rc2", "periodic"),
        ("periodic", "rc3", "periodic"),
        ("channel", "rc1", "channel"),
        ("channel", "rc2", "channel"),
        ("channel", "rc3", "channel"),
    ),
)
def test_frozen_release_identity_ignores_future_current_runtime_fields(
    release_runtimes,
    family,
    generation,
    manifest_key,
):
    periodic, channel = release_runtimes
    runtime = periodic if family == "periodic" else channel
    metadata = runtime.identity().to_metadata()
    expected = json.loads((_FIXTURE_ROOT / "manifest.json").read_text())[
        "releases"
    ][generation][manifest_key]["functional_runtime_identity_sha256"]
    observed = legacy_runtime_identity_sha256(
        metadata,
        family=family,
        source_release_generation=generation,
    )
    mutated = copy.deepcopy(metadata)
    mutated["execution"]["functional_runtime"][
        "future_current_only_field"
    ] = {"must_not_rewrite_released_identity": True}
    after_growth = legacy_runtime_identity_sha256(
        mutated,
        family=family,
        source_release_generation=generation,
    )
    assert observed == expected
    assert after_growth == expected
