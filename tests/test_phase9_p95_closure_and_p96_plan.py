"""Static closure and consumer-contract gates for P9.5--P9.6."""

from __future__ import annotations

import json
from pathlib import Path

from pssolver.functional.periodic_activity import (
    PERIODIC_ACTIVITY_DETERMINISTIC_REPLAY,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _record(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def test_p95_h100_closure_is_complete_and_authorizes_only_p96():
    record = _record("phase_9_p95_h100_closure.json")
    assert record["classification"] == "PASS_P9_5_H100_BATCH_ONE_FORWARD_VJP_MEMORY"
    assert record["source_commit"] == "bdf9268a8b668eb79510dc0c51bae587142d26a8"
    assert record["scope"]["batch_sizes_qualified"] == [1]
    assert record["h100"]["state"] == "COMPLETED"
    assert record["h100"]["exit_code"] == "0:0"
    assert record["h100"]["profile_count"] == 18
    assert record["correctness"]["deterministic_replay"] == "bitwise"
    assert record["stable_gradient_validator_v2"][
        "all_six_gradient_paths_finite_nonzero_and_passed"
    ] is True
    assert record["archive"]["checksum_passed"] == 43
    assert record["authorization"] == {
        "p9_5_complete": True,
        "p9_6_authorized": True,
        "p9_7_authorized": False,
        "p9_8_authorized": False,
        "larger_batch_authorized": False,
        "production_default_changed": False,
        "verbatim_pdf_regenerated": False,
    }


def test_p95_h100_evidence_promotes_periodic_cuda_replay_declaration():
    assert PERIODIC_ACTIVITY_DETERMINISTIC_REPLAY == "bitwise"


def test_p96_preserves_dependency_and_ownership_boundaries():
    record = _record("phase_9_p96_independent_consumer_plan.json")
    assert record["status"] == "authorized_pending_independent_consumer_implementation"
    assert record["ownership"]["dependency_direction"] == (
        "independent_consumer_to_pssolver_only"
    )
    assert record["ownership"]["pssolver_imports_consumer"] is False
    assert record["provider_api"]["module"] == "pssolver.functional"
    assert record["provider_api"]["forbidden_private_access"] is True
    assert record["provider_api"]["legacy_fallback"] is False
    assert record["adapter_mapping"]["control"] == (
        "single activity tensor <-> {'activity': tensor}"
    )
    assert record["adapter_mapping"]["implicit_tensor_conversion"] is False


def test_p96_is_batch_one_periodic_consumer_not_a_science_campaign():
    record = _record("phase_9_p96_independent_consumer_plan.json")
    exercise = record["consumer_exercise"]
    assert exercise["geometry"] == "periodic_box"
    assert exercise["batch_size"] == 1
    assert exercise["checkpoint_strides"] == [1, 2, 4]
    assert exercise["optimizer_iteration_required"] is False
    assert exercise["scientific_control_claim"] is False
    assert record["h100"]["grid"]["shape"] == [128, 128, 32]
    assert record["authorization"]["p9_7_authorized"] is False
    assert record["authorization"]["larger_batch_authorized"] is False


def test_verbatim_archive_source_tracks_closure_and_p96_without_rendering_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    for name in (
        "phase_9_p95_h100_closure.md",
        "phase_9_p95_h100_closure.json",
        "phase_9_p96_independent_consumer_plan.md",
        "phase_9_p96_independent_consumer_plan.json",
    ):
        assert f'"{name}"' in source
    assert _record("phase_9_p95_h100_closure.json")["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
    assert _record("phase_9_p96_independent_consumer_plan.json")["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
