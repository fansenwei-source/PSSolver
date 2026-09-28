"""Static closure record for P9.1 functional prerequisites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p91_functional_prerequisites.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p91_record_binds_p90_and_closes_only_prerequisites():
    record = _record()

    assert record["phase"] == "P9.1"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_1_FUNCTIONAL_PREREQUISITES"
    )
    baseline = record["baseline"]
    assert baseline["commit"] == (
        "30c1d4c92045f0cb5db6ef58c3e0d49feb598feb"
    )
    assert _sha256(ROOT / baseline["p9_0_record"]) == baseline[
        "p9_0_record_sha256"
    ]
    assert baseline["p9_0_complete"] is True
    assert record["authorization"] == {
        "p9_1_complete": True,
        "p9_2_implementation_authorized": False,
        "later_phase_9_slices_authorized": False,
        "h100_authorized": False,
        "production_default_changed": False,
        "verbatim_pdf_regenerated": False,
    }


def test_p91_records_the_two_periodic_zero_mode_branches_without_new_physics():
    friction = _record()["periodic_friction"]

    assert friction["publicly_connected"] is True
    assert friction["pressure_gauge"] == "zero_mean"
    assert friction["valid_zero_mean_pair"]["friction"] == 0.0
    assert friction["valid_friction_pair"]["friction_constraint"] == (
        "finite_and_strictly_positive"
    )
    assert friction["default_changed"] is False
    assert friction["equation_or_operator_changed"] is False
    assert friction["plane_or_channel_compiler_broadened"] is False


def test_p91_records_explicit_planar_embedding_without_a_second_q_theory():
    contract = _record()["z_invariant_contract"]

    assert contract["nz_one_public_step_qualified_on_cpu"] is True
    assert contract["multi_plane_invariant_subspace_qualified_on_cpu"] is True
    assert contract["planar_components"] == ["Qxx", "Qxy", "Qyy"]
    assert contract["three_dimensional_components"] == [
        "Qxx",
        "Qxy",
        "Qxz",
        "Qyy",
        "Qyz",
    ]
    assert contract["out_of_plane_components_exactly_zero"] is True
    assert contract["projection_is_fail_closed_not_averaging"] is True
    assert contract["separate_two_dimensional_q_theory_added"] is False


def test_p91_functional_surface_is_provisional_declarative_and_fail_closed():
    functional = _record()["functional_declarations"]

    assert functional["namespace"] == "pssolver.functional"
    assert functional["api_version"] == "0.1-provisional"
    assert functional["stable_package_root_exported"] is False
    assert functional["runtime_protocol_declared"] is True
    assert functional["factory_protocol_declared"] is True
    assert functional["first_required_batch_size"] == 1
    assert functional["larger_batches_claimed"] is False
    assert functional["observation_time"] == (
        "input_state_under_current_control"
    )
    assert functional["default_capabilities_fail_closed"] is True
    assert functional["functional_runtime_implemented"] is False
    assert functional["functional_timestep_implemented"] is False
    assert functional["gradient_or_adjoint_implemented"] is False

    package_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "pssolver" / "functional").glob("*.py"))
    )
    assert "pssolver.control" not in package_sources
    assert "PSSolver-Control" not in package_sources


def test_future_archive_lists_p91_without_regenerating_the_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    p90 = '"phase_9_p90_control_contract_freeze.json"'
    p91_md = '"phase_9_p91_functional_prerequisites.md"'
    p91_json = '"phase_9_p91_functional_prerequisites.json"'

    assert source.count(p91_md) == 1
    assert source.count(p91_json) == 1
    assert source.index(p90) < source.index(p91_md) < source.index(p91_json)
    assert _record()["local_verification"]["verbatim_pdf_regenerated"] is False
