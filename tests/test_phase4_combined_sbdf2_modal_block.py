"""Phase 4.5 combined SBDF2/modal-block canary qualification gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import pssolver
import pssolver.integrators as integrators
from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec
from pssolver.experimental.modal_block_checkpoint import (
    COMPLETE_NAME,
    MANIFEST_NAME,
    load_two_component_reference_checkpoint,
    save_two_component_reference_checkpoint,
)
from pssolver.experimental.modal_block_reference import (
    SBDF2_OPERATIONS,
    STARTUP_OPERATIONS,
    TwoComponentReferenceState,
    TwoComponentSBDF2CanaryModel,
    TwoComponentSBDF2ReferenceStepper,
)
from pssolver.operators.modal_block_reference import (
    TwoComponentPeriodicReactionDiffusionReference,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p45_combined_sbdf2_modal_block.json"
)
RUNTIME_PATHS = tuple((ROOT / "pssolver" / "runtime").glob("*.py"))


def _model() -> TwoComponentSBDF2CanaryModel:
    operator = TwoComponentModalOperatorSpec(
        component_order=("u", "v"),
        diffusion=(0.12, 0.2),
        coupling=((-0.3, 0.4), (-0.2, -0.1)),
    )
    implicit = TwoComponentPeriodicReactionDiffusionReference(
        operator_spec=operator,
        point_count=32,
        length=2.0 * math.pi,
        initial_mode=2,
        initial_amplitudes=(0.8, -0.35),
    )
    return TwoComponentSBDF2CanaryModel(
        implicit_model=implicit,
        explicit_growth_rate=0.07,
    )


def _stepper(dt: float = 0.01) -> TwoComponentSBDF2ReferenceStepper:
    return TwoComponentSBDF2ReferenceStepper(
        model=_model(),
        spec=IntegratorSpec.sbdf2(dt=dt),
        dtype=torch.float64,
        device="cpu",
        implementation="closed_form_2x2",
    )


def _rewrite_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    data = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (checkpoint / MANIFEST_NAME).write_bytes(data)
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    (checkpoint / COMPLETE_NAME).write_text(
        f"manifest_sha256={digest}\n",
        encoding="utf-8",
    )


def test_combined_canary_executes_frozen_startup_and_sbdf2_order():
    stepper = _stepper()
    initial = stepper.initial_state(refresh_interval=2)
    startup = stepper.step(initial)
    assert startup.scheme_used is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    assert startup.startup_step is True
    assert startup.operation_order == STARTUP_OPERATIONS
    assert startup.state.history is not None
    second = stepper.step(startup.state)
    assert second.scheme_used is IntegratorScheme.SBDF2
    assert second.startup_step is False
    assert second.refreshed is True
    assert second.operation_order == SBDF2_OPERATIONS


def test_combined_workspace_is_bounded_and_reused_by_identity():
    stepper = _stepper()
    workspace = stepper.workspace
    identities = (
        id(workspace.current_explicit_rhs),
        id(workspace.assembled_implicit_rhs),
        id(workspace.modal_solve.output),
    )
    state = stepper.run(steps=4)
    assert state.completed_steps == 4
    assert identities == (
        id(workspace.current_explicit_rhs),
        id(workspace.assembled_implicit_rhs),
        id(workspace.modal_solve.output),
    )
    assert workspace.allocated_tensor_count == 7
    assert workspace.to_metadata()["bounded"] is True
    assert state.history is not None
    assert state.native_spectrum.data_ptr() != (
        state.history.previous_evolved_native_spectrum.data_ptr()
    )


def test_combined_sbdf2_reaches_frozen_second_order_threshold():
    model = _model()
    exact = model.exact_physical(
        0.4,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    errors = []
    linf = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        state = _stepper(dt).run(steps=round(0.4 / dt))
        difference = state.physical - exact
        errors.append(
            float(
                torch.linalg.vector_norm(difference)
                / math.sqrt(difference.numel())
            )
        )
        linf.append(float(torch.max(torch.abs(difference))))
    orders = [
        math.log(coarse / fine, 2.0)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]
    assert min(orders) >= 1.8
    assert all(math.isfinite(value) for value in (*errors, *linf))


def test_continuous_and_in_memory_split_are_byte_identical():
    stepper = _stepper()
    continuous = stepper.run(steps=40, refresh_interval=7)
    split = stepper.run(steps=17, refresh_interval=7)
    restarted = stepper.run(steps=23, state=split)
    assert torch.equal(restarted.physical, continuous.physical)
    assert torch.equal(restarted.native_spectrum, continuous.native_spectrum)
    assert restarted.history is not None and continuous.history is not None
    assert torch.equal(
        restarted.history.previous_evolved_native_spectrum,
        continuous.history.previous_evolved_native_spectrum,
    )
    assert torch.equal(
        restarted.history.previous_explicit_native_spectral_rhs,
        continuous.history.previous_explicit_native_spectral_rhs,
    )


def test_checkpoint_roundtrip_and_restart_are_byte_identical(tmp_path):
    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    split = stepper.run(steps=17, refresh_interval=7)
    manifest = save_two_component_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=split,
    )
    assert manifest["workflow"] == "two_component_sbdf2_modal_block_canary"
    loaded = load_two_component_reference_checkpoint(
        checkpoint,
        stepper=stepper,
    )
    continuous = stepper.run(steps=40, refresh_interval=7)
    restarted = stepper.run(steps=23, state=loaded)
    assert torch.equal(restarted.physical, continuous.physical)
    assert torch.equal(restarted.native_spectrum, continuous.native_spectrum)
    assert restarted.history is not None and continuous.history is not None
    assert torch.equal(
        restarted.history.previous_evolved_native_spectrum,
        continuous.history.previous_evolved_native_spectrum,
    )
    assert (checkpoint / COMPLETE_NAME).is_file()


def test_checkpoint_rejects_tamper_missing_complete_and_changed_dt(tmp_path):
    stepper = _stepper()
    source = stepper.run(steps=3)
    tampered = tmp_path / "tampered"
    save_two_component_reference_checkpoint(
        tampered,
        stepper=stepper,
        state=source,
    )
    path = tampered / "previous_explicit_native_spectral_rhs.npy"
    array = np.load(path, allow_pickle=False)
    array.flat[0] += 1.0
    np.save(path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        load_two_component_reference_checkpoint(tampered, stepper=stepper)

    incomplete = tmp_path / "incomplete"
    save_two_component_reference_checkpoint(
        incomplete,
        stepper=stepper,
        state=source,
    )
    (incomplete / COMPLETE_NAME).unlink()
    with pytest.raises(ValueError, match="COMPLETE"):
        load_two_component_reference_checkpoint(incomplete, stepper=stepper)

    incompatible = tmp_path / "incompatible"
    save_two_component_reference_checkpoint(
        incompatible,
        stepper=stepper,
        state=source,
    )
    with pytest.raises(ValueError, match="identity"):
        load_two_component_reference_checkpoint(
            incompatible,
            stepper=_stepper(0.02),
        )


def test_checkpoint_rejects_nonfinite_tensor_with_rewritten_hash(tmp_path):
    import hashlib

    stepper = _stepper()
    checkpoint = tmp_path / "checkpoint"
    save_two_component_reference_checkpoint(
        checkpoint,
        stepper=stepper,
        state=stepper.run(steps=3),
    )
    manifest = json.loads(
        (checkpoint / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    path = checkpoint / "native_spectrum.npy"
    array = np.load(path, allow_pickle=False)
    array.flat[0] = np.nan
    np.save(path, array, allow_pickle=False)
    record = manifest["tensors"]["native_spectrum"]
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["size_bytes"] = path.stat().st_size
    _rewrite_manifest(checkpoint, manifest)
    with pytest.raises(ValueError, match="non-finite"):
        load_two_component_reference_checkpoint(checkpoint, stepper=stepper)


def test_failed_stage_preserves_state_and_history():
    stepper = _stepper()
    state = stepper.run(steps=4)
    assert state.history is not None
    snapshots = (
        state.physical.clone(),
        state.native_spectrum.clone(),
        state.history.previous_evolved_native_spectrum.clone(),
        state.history.previous_explicit_native_spectral_rhs.clone(),
    )

    def fail(operation: str) -> None:
        if operation == "solve_implicit_operator":
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        stepper.step(state, stage_observer=fail)
    assert torch.equal(state.physical, snapshots[0])
    assert torch.equal(state.native_spectrum, snapshots[1])
    assert torch.equal(
        state.history.previous_evolved_native_spectrum,
        snapshots[2],
    )
    assert torch.equal(
        state.history.previous_explicit_native_spectral_rhs,
        snapshots[3],
    )


def test_combined_canary_metadata_and_import_surface_are_opt_in():
    metadata = _stepper().to_metadata()
    assert metadata["identity"] == (
        "two_component_sbdf2_modal_block_reference_stepper"
    )
    assert metadata["workspace"]["allocated_tensor_count"] == 7
    assert metadata["model"]["split"] == {
        "implicit": "two_component_reaction_diffusion_modal_block",
        "explicit": "scalar_identity_growth",
    }
    assert metadata["connection"] == {
        "runtime_state": False,
        "step_program": False,
        "plane": False,
        "production_checkpoint": False,
        "package_root_export": False,
    }
    names = {
        "TwoComponentSBDF2CanaryModel",
        "TwoComponentSBDF2ReferenceStepper",
        "TwoComponentReferenceState",
    }
    assert names.isdisjoint(pssolver.__all__)
    assert names.isdisjoint(integrators.__all__)
    for path in RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "experimental.modal_block_reference" not in source
        assert "experimental.modal_block_checkpoint" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_canary_binds_concrete_device_and_remains_finite():
    stepper = TwoComponentSBDF2ReferenceStepper(
        model=_model(),
        spec=IntegratorSpec.sbdf2(dt=0.01),
        dtype=torch.float64,
        device="cuda",
        implementation="closed_form_2x2",
    )
    state = stepper.run(steps=40, refresh_interval=7)
    torch.cuda.synchronize()
    assert state.physical.device == stepper.operator.device
    assert state.physical.device.index == torch.cuda.current_device()
    assert bool(torch.isfinite(state.physical).all())
    assert stepper.workspace.allocated_tensor_count == 7


def test_phase4_p45_machine_record_matches_opt_in_contract():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["status"] == "P4_5_LOCAL_COMPLETE_OPT_IN_REFERENCE"
    assert record["p4_4_authority"]["classification"] == (
        "PASS_P4_4_MODAL_BLOCK_H100_CONTRACT_EQUIVALENT"
    )
    assert record["integrator"]["scheme"] == "sbdf2"
    assert record["integrator"]["minimum_observed_l2_order"] > 2.0
    assert record["workspace"]["allocated_tensor_count"] == 7
    assert record["restart"]["checkpoint_byte_identical"] is True
    assert record["eligibility"] == {
        "p4_5_local_complete": True,
        "p4_6_authorized": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
