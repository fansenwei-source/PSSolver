from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s18_r5_channel_pcg_control.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_r5_record_freezes_the_private_four_path_contract() -> None:
    record = _record()
    assert record["plan_step"] == "S-18"
    assert record["batch"] == "R5"
    assert record["governance"] == {
        "decision": "D-0018",
        "design_review": "R-0054",
        "implementation_commit": (
            "318e634eb09de27cb683c31c4860ab34761e8e44"
        ),
        "implementation_parent": (
            "d84e883df58af92268dc55f0b02a78b3b5cffd57"
        ),
        "qualification_support_commit": (
            "0238deabc280193e79aeaeb5c0074886bb7b6e1b"
        ),
        "comparison_baseline": "v0.3.0",
        "nonzero_tolerance_authorized": False,
    }
    implementation = record["implementation"]
    assert implementation["path_specific_tensor_arithmetic_retained"] is True
    assert implementation["public_reexport_added"] is False
    assert len(implementation["paths"]) == 4


def test_r5_record_authorizes_no_external_or_gpu_action() -> None:
    record = _record()
    assert record["frozen_contract"]["nonzero_comparison_tolerance"] is None
    assert record["frozen_contract"][
        "d0013_memory_exception_extended_to_v0_4"
    ] is False
    assert record["scope"] == {
        "public_api_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "provider_identity_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
    }
    assert record["authorization"] == {
        "local_implementation": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
