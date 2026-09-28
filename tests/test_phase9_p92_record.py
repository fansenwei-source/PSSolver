"""Static closure record for the P9.2 periodic functional runtime."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p92_periodic_functional_runtime.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p92_record_closes_only_the_periodic_batch_one_slice():
    record = _record()

    assert record["phase"] == "P9.2"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_2_PERIODIC_ACTIVITY_FUNCTIONAL_RUNTIME"
    )
    assert record["baseline"]["commit"] == (
        "dc7177fb95bc94797aa995553d986a9229b14378"
    )
    assert record["scope"] == {
        "model": "complete_stress_beris_edwards",
        "geometry": "periodic_box",
        "runtime_path": "periodic_spectral",
        "batch_sizes": [1],
        "pointwise_execution": "eager",
        "spectral_refresh": "disabled",
        "production_default_changed": False,
        "stable_package_root_changed": False,
    }
    assert record["authorization"]["p9_2_complete"] is True
    assert record["authorization"]["p9_3_implementation_authorized"] is False
    assert record["authorization"]["h100_authorized"] is False


def test_p92_record_makes_all_persistent_state_and_control_semantics_explicit():
    record = _record()

    assert record["state"]["components"] == ["q_physical", "q_spectral"]
    assert record["state"]["all_persistent_next_step_values_included"] is True
    assert record["state"]["caller_tensor_mutation"] is False
    assert record["control"]["equation_term"] == "div(beta * alpha * Q)"
    assert record["control"]["injection_order"] == (
        "form_beta_alpha_q_product_before_divergence"
    )
    assert record["control"]["implicit_clipping"] is False
    assert record["control"]["parameter_registry_mutation"] is False
    assert record["observations"]["time_alignment"] == (
        "input_state_under_current_control"
    )
    assert record["observations"]["terminal_without_control"] == ["Q"]


def test_p92_record_does_not_overclaim_later_qualifications():
    capabilities = _record()["capabilities"]

    assert capabilities["pure_step"] is True
    assert capabilities["differentiability"] == "torch_autograd"
    assert capabilities["inner_solve_gradient"] == (
        "direct_periodic_fourier_autograd"
    )
    assert capabilities["deterministic_replay"] == "not_qualified"
    assert capabilities["durable_checkpoint_bridge"] is False
    assert capabilities["explicit_jvp"] is False
    assert capabilities["explicit_vjp"] is False


def test_p92_runtime_has_no_dependency_on_legacy_or_external_control_code():
    source = (ROOT / "pssolver" / "functional" / "periodic_activity.py").read_text(
        encoding="utf-8"
    )

    assert "pssolver.control" not in source
    assert "PSSolver-Control" not in source
    assert _record()["reuse"]["legacy_control_oracle_changed"] is False
    assert _record()["reuse"]["external_control_dependency_added"] is False


def test_future_archive_lists_p92_without_regenerating_the_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    p91 = '"phase_9_p91_functional_prerequisites.json"'
    p92_md = '"phase_9_p92_periodic_functional_runtime.md"'
    p92_json = '"phase_9_p92_periodic_functional_runtime.json"'

    assert source.count(p92_md) == 1
    assert source.count(p92_json) == 1
    assert source.index(p91) < source.index(p92_md) < source.index(p92_json)
    assert _record()["qualification"]["verbatim_pdf_regenerated"] is False
