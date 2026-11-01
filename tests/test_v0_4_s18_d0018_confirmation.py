"""Forward contract for the confirmed D-0018 S-18 design choices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT / "notes/PSSolver_v0_3_0_s03_successor_93_d0018_confirmation.json"
)
DESIGN = ROOT / "notes/PSSolver_v0_4_0_s18_r2_r4_r5_r1_design_proposal.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_93_binds_confirmed_d0018_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 93
    assert record["classification"] == "D0018_CONFIRMED_PENDING_R0054"
    assert record["decision"] == {
        "id": "D-0018",
        "status": "confirmed",
        "sync_commit": "4cf34051cd5d4f47801eaabf0bdc7cd8490d6367",
        "sha256": (
            "38cfe71b63efcca6ef47a759d5955bd332266cf4a65cef6e61635c43bfd1a627"
        ),
        "choices": {
            "Q-S18-01": "A",
            "Q-S18-02": "A",
            "Q-S18-03": "A",
            "Q-S18-04": "A",
            "Q-S18-05": "A",
        },
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_confirmed_design_keeps_implementation_closed_until_r0054_pass() -> None:
    design = _read(DESIGN)
    assert design["classification"] == (
        "DESIGN_ONLY_D0018_CONFIRMED_AWAITING_R0054_REVIEW"
    )
    assert design["governance"]["design_review_request"] == "R-0054"
    assert design["governance"]["superseded_preconfirmation_request"] == "R-0053"
    assert {item["id"]: item["selected_option"] for item in design["confirmed_choices"]} == {
        "Q-S18-01": "A",
        "Q-S18-02": "A",
        "Q-S18-03": "A",
        "Q-S18-04": "A",
        "Q-S18-05": "A",
    }
    assert design["implementation"]["started"] is False
    assert all(value is False for value in design["authorization"].values())


def test_v60_remains_claimed_at_the_reviewed_scope_fix() -> None:
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-60")
    assert issue["status"] in {"claimed_fixed", "verified"}
    assert issue["fixed_in"] == "f5dbdbef43d1aa16e9c7638093df5c5c5ae58170"
    assert issue["target_release"] == "0.4 S-18"


def test_d0018_confirmation_changes_no_runtime_or_authorization() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
