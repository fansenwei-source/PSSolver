"""RC4.4.1 analysis-only pressure-contract adjudication gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc441_pressure_adjudication.json"
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc440_pressure_contract_plan.json"
DIAGNOSTIC = ROOT / "benchmarks/diagnose_channel_pressure_contract.py"
RC444 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc444_pressure_metadata_identity_connection.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc441_record_binds_the_plan_and_reproducible_diagnostic():
    record = _record()

    assert record["classification"] == (
        "PASS_RC4_4_1_ACTIVE_MODE_RMS_AND_SCOPED_CAPABILITY_CONTRACT"
    )
    assert record["identity"]["rc4_4_0_plan_sha256"] == _sha256(PLAN)
    observed = _sha256(DIAGNOSTIC)
    historical = record["identity"]["diagnostic_sha256"]
    assert historical != observed
    successor = json.loads(RC444.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]["benchmarks/diagnose_channel_pressure_contract.py"]
    assert successor["historical_sha256"] == historical
    assert successor["current_sha256"] == observed
    assert record["completion"] == {
        "analysis_complete": True,
        "B25_formula_selected": True,
        "B28_truth_table_complete": True,
        "B28_source_repair_complete": False,
        "B29_policy_reconfirmed": True,
        "eligible_for_rc4_4_2": True,
        "solver_source_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "installed_wheel_qualified": False,
        "H100_qualified": False,
        "rc4_4_complete": False,
    }
