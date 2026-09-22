"""Phase 4.3 convergence, checkpoint, restart, and negative gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import pssolver
import pssolver.integrators as integrators
from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.reference_checkpoint import (
    COMPLETE_NAME,
    FORMAT_IDENTITY,
    MANIFEST_NAME,
    load_scalar_reference_checkpoint,
    save_scalar_reference_checkpoint,
)
from pssolver.integrators.reference_qualification import (
    TEMPORAL_STEP_RATIO_LADDER,
    measure_temporal_convergence,
)
from pssolver.integrators.scalar_reference import (
    ScalarPeriodicReactionDiffusion,
    ScalarPeriodicReferenceStepper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P43_RECORD_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p43_convergence_restart.json"
)
RUNTIME_PATHS = tuple((PROJECT_ROOT / "pssolver" / "runtime").glob("*.py"))


def _model(**updates) -> ScalarPeriodicReactionDiffusion:
    values = {
        "point_count": 32,
        "length": 2.0 * math.pi,
        "diffusivity": 0.2,
        "reaction_rate": 0.15,
        "initial_mode": 3,
        "initial_amplitude": 0.75,
    }
    values.update(updates)
    return ScalarPeriodicReactionDiffusion(**values)


def _stepper(
    *,
    dt: float = 0.01,
    scheme: IntegratorScheme = IntegratorScheme.SBDF2,
    model: ScalarPeriodicReactionDiffusion | None = None,
) -> ScalarPeriodicReferenceStepper:
    spec = (
        IntegratorSpec.sbdf2(dt=dt)
        if scheme is IntegratorScheme.SBDF2
        else IntegratorSpec.projected_semi_implicit_euler(dt=dt)
    )
    return ScalarPeriodicReferenceStepper(model=model or _model(), spec=spec)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _rewrite_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    data = _canonical_json_bytes(manifest)
    (checkpoint / MANIFEST_NAME).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (checkpoint / COMPLETE_NAME).write_text(
        f"manifest_sha256={digest}\n",
        encoding="utf-8",
    )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().numpy().tobytes()


@pytest.mark.parametrize(
    ("scheme", "minimum_order"),
    [
        (IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER, 0.9),
        (IntegratorScheme.SBDF2, 1.8),
    ],
)
def test_frozen_temporal_ladder_passes_required_order(scheme, minimum_order):
    result = measure_temporal_convergence(model=_model(), scheme=scheme)
    assert TEMPORAL_STEP_RATIO_LADDER == (1.0, 0.5, 0.25, 0.125)
    assert tuple(sample.dt for sample in result.samples) == pytest.approx(
        (0.04, 0.02, 0.01, 0.005)
    )
    assert tuple(sample.steps for sample in result.samples) == (10, 20, 40, 80)
    assert all(sample.final_time == pytest.approx(0.4) for sample in result.samples)
    assert all(sample.finite for sample in result.samples)
    assert all(
        coarse.l2_error > fine.l2_error
        for coarse, fine in zip(
            result.samples[:-1],
            result.samples[1:],
            strict=True,
        )
    )
    assert all(
        coarse.linf_error > fine.linf_error
        for coarse, fine in zip(
            result.samples[:-1],
            result.samples[1:],
            strict=True,
        )
    )
    assert result.minimum_required_order == minimum_order
    assert result.minimum_observed_order >= minimum_order
    assert result.passed is True


def test_frozen_temporal_ladder_observed_orders_match_local_oracle():
    euler = measure_temporal_convergence(
        model=_model(),
        scheme=IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER,
    )
    sbdf2 = measure_temporal_convergence(
        model=_model(),
        scheme=IntegratorScheme.SBDF2,
    )
    assert euler.pairwise_l2_orders == pytest.approx(
        (0.9753185405, 0.9874434658, 0.9936660939),
        abs=2e-9,
    )
    assert sbdf2.pairwise_l2_orders == pytest.approx(
        (2.0448688430, 2.0191124362, 2.0087612216),
        abs=2e-9,
    )
    assert euler.to_metadata()["passed"] is True
    assert sbdf2.to_metadata()["passed"] is True


@pytest.mark.parametrize(
    ("base_dt", "final_time", "message"),
    [
        (0.0, 0.4, "base_dt"),
        (0.04, 0.0, "final_time"),
        (0.03, 0.4, "integer multiple"),
    ],
)
def test_temporal_ladder_rejects_invalid_time_contract(
    base_dt,
    final_time,
    message,
):
    with pytest.raises(ValueError, match=message):
        measure_temporal_convergence(
            model=_model(),
            scheme=IntegratorScheme.SBDF2,
            base_dt=base_dt,
            final_time=final_time,
        )


def test_checkpoint_roundtrip_preserves_state_history_and_identity(tmp_path):
    stepper = _stepper()
    state = stepper.run(steps=5, refresh_interval=4)
    checkpoint = tmp_path / "checkpoint"
    manifest = save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=state,
    )
    loaded = load_scalar_reference_checkpoint(checkpoint, stepper=stepper)
    assert manifest["format"] == FORMAT_IDENTITY
    assert manifest["runtime"]["phase"] == "multistep"
    assert manifest["history"] == {
        "present": True,
        "depth": 1,
        "source_completed_steps": 4,
        "dt": 0.01,
    }
    assert (checkpoint / COMPLETE_NAME).is_file()
    assert torch.equal(loaded.physical, state.physical)
    assert torch.equal(loaded.native_spectrum, state.native_spectrum)
    assert loaded.completed_steps == state.completed_steps
    assert loaded.refresh_interval == state.refresh_interval
    assert loaded.refresh_step_count == state.refresh_step_count
    assert loaded.refresh_count == state.refresh_count
    assert loaded.history is not None
    assert state.history is not None
    assert torch.equal(
        loaded.history.previous_evolved_native_spectrum,
        state.history.previous_evolved_native_spectrum,
    )
    assert torch.equal(
        loaded.history.previous_explicit_native_spectral_rhs,
        state.history.previous_explicit_native_spectral_rhs,
    )


def test_continuous_and_split_restart_are_byte_identical(tmp_path):
    stepper = _stepper()
    continuous = stepper.run(steps=12, refresh_interval=4)
    split = stepper.run(steps=5, refresh_interval=4)
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=split,
    )
    restarted = load_scalar_reference_checkpoint(checkpoint, stepper=stepper)
    restarted = stepper.run(steps=7, state=restarted)
    assert _tensor_bytes(restarted.physical) == _tensor_bytes(continuous.physical)
    assert _tensor_bytes(restarted.native_spectrum) == _tensor_bytes(
        continuous.native_spectrum
    )
    assert restarted.completed_steps == continuous.completed_steps
    assert restarted.refresh_step_count == continuous.refresh_step_count
    assert restarted.refresh_count == continuous.refresh_count
    assert restarted.history is not None
    assert continuous.history is not None
    assert _tensor_bytes(
        restarted.history.previous_evolved_native_spectrum
    ) == _tensor_bytes(continuous.history.previous_evolved_native_spectrum)
    assert _tensor_bytes(
        restarted.history.previous_explicit_native_spectral_rhs
    ) == _tensor_bytes(
        continuous.history.previous_explicit_native_spectral_rhs
    )


def test_checkpoint_refuses_existing_target_without_overwrite(tmp_path):
    stepper = _stepper()
    target = tmp_path / "checkpoint"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        save_scalar_reference_checkpoint(
            target,
            stepper=stepper,
            state=stepper.run(steps=2),
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_checkpoint_rejects_missing_complete_marker(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    (checkpoint / COMPLETE_NAME).unlink()
    with pytest.raises(ValueError, match="COMPLETE marker is missing"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_checkpoint_rejects_missing_history_tensor(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    (checkpoint / "previous_evolved_native_spectrum.npy").unlink()
    with pytest.raises(ValueError, match="is missing"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_checkpoint_rejects_tampered_history_tensor(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    path = checkpoint / "previous_explicit_native_spectral_rhs.npy"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size mismatch"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_checkpoint_rejects_nonfinite_tensor_even_with_updated_hash(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    manifest = json.loads((checkpoint / MANIFEST_NAME).read_text(encoding="utf-8"))
    path = checkpoint / "previous_evolved_native_spectrum.npy"
    array = np.load(path, allow_pickle=False)
    array[1] = complex(math.nan, 0.0)
    np.save(path, array, allow_pickle=False)
    record = manifest["tensors"]["previous_evolved_native_spectrum"]
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["size_bytes"] = path.stat().st_size
    record["finite"] = True
    _rewrite_manifest(checkpoint, manifest)
    with pytest.raises(ValueError, match="is non-finite"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_checkpoint_rejects_changed_dt_before_advance(tmp_path):
    original = _stepper(dt=0.01)
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=original,
        state=original.run(steps=2),
    )
    with pytest.raises(ValueError, match="integrator identity mismatch"):
        load_scalar_reference_checkpoint(
            checkpoint,
            stepper=_stepper(dt=0.02),
        )


def test_checkpoint_rejects_different_integrator_before_advance(tmp_path):
    original = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=original,
        state=original.run(steps=2),
    )
    with pytest.raises(ValueError, match="integrator identity mismatch"):
        load_scalar_reference_checkpoint(
            checkpoint,
            stepper=_stepper(
                scheme=IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            ),
        )


def test_checkpoint_rejects_unsupported_schema_version(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    manifest = json.loads((checkpoint / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    _rewrite_manifest(checkpoint, manifest)
    with pytest.raises(ValueError, match="schema version is unsupported"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_checkpoint_rejects_incompatible_model_discretization(tmp_path):
    original = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=original,
        state=original.run(steps=2),
    )
    incompatible = _stepper(model=_model(point_count=64))
    with pytest.raises(ValueError, match="model/discretization identity mismatch"):
        load_scalar_reference_checkpoint(checkpoint, stepper=incompatible)


def test_checkpoint_rejects_stale_history_clock(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_scalar_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=2),
    )
    manifest = json.loads((checkpoint / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["history"]["source_completed_steps"] = 0
    _rewrite_manifest(checkpoint, manifest)
    with pytest.raises(ValueError, match="immediately precede"):
        load_scalar_reference_checkpoint(checkpoint, stepper=stepper)


def test_p43_types_are_direct_import_only_and_disconnected():
    names = {
        "TemporalErrorSample",
        "TemporalConvergenceResult",
        "measure_temporal_convergence",
        "save_scalar_reference_checkpoint",
        "load_scalar_reference_checkpoint",
    }
    assert names.isdisjoint(pssolver.__all__)
    assert names.isdisjoint(integrators.__all__)
    for path in RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "pssolver.integrators.reference_checkpoint" not in source
        assert "pssolver.integrators.reference_qualification" not in source


def test_phase4_p43_machine_record_matches_qualified_contract():
    record = json.loads(P43_RECORD_PATH.read_text(encoding="utf-8"))
    assert record["status"] == "P4_3_LOCAL_COMPLETE_REFERENCE_ONLY"
    assert record["baseline_commit"] == (
        "2393a59bb9464ad02705dc257a2e067a7fa35104"
    )
    assert record["convergence"]["ladder"] == [1.0, 0.5, 0.25, 0.125]
    assert record["convergence"]["euler"]["minimum_required_order"] == 0.9
    assert record["convergence"]["euler"]["passed"] is True
    assert record["convergence"]["sbdf2"]["minimum_required_order"] == 1.8
    assert record["convergence"]["sbdf2"]["passed"] is True
    assert record["restart"]["continuous_split_byte_identical"] is True
    assert record["restart"]["negative_gates_passed"] == 8
    assert record["connection"] == {
        "runtime_state_connected": False,
        "step_program_connected": False,
        "plane_connected": False,
        "plane_checkpoint_v1_changed": False,
        "package_root_exported": False,
        "integrators_package_exported": False,
    }
    assert record["validation"] == {
        "focused_tests_passed": 90,
        "complete_tests_passed": 1746,
        "subtests_passed": 8,
        "failures": 0,
        "archive_sources_verified": 56,
        "git_diff_check_passed": True,
    }
    assert record["eligibility"] == {
        "p4_3_complete": True,
        "p4_4_authorized": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
