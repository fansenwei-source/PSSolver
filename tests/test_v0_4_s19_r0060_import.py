"""Repository-only, forward-resolving R-0060 preparation review record."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_120_r0060_import.json"


def _read():
    return json.loads(RECORD.read_text())


def test_r0060_binds_interim_review_and_readonly_archive():
    review = _read()["review"]
    assert review["classification"] == "INTERIM_PASS_S19_LOCAL_PREPARATION"
    assert review["result_sha256"] == "cb6412b10ad588daf215bdbcc6dcfb8e424c8524ba017f5cf94ae9364225caeb"
    assert review["archive"]["checksums_entries"] == 130
    assert review["archive"]["checksums_sha256"] == "8db9ff91b791b210c6700f9f0987ab2caf2ff308f6208bf0bb246070b1fa3b73"


def test_r0060_snapshot_closes_only_v62_and_preserves_fix():
    record = _read()
    assert record["registry_snapshot"] == {
        "V-62": {"status": "verified", "verified_by": "R-0060",
                 "fixed_in": "949970e5f32126ff50f17ef812cfe288f8eb9729"},
        "V-63": {"status": "open", "target_release": "0.4 S-20"},
    }
    assert record["new_findings"] == []
    assert record["git_identity"]["reviewed_support"] == "141ad2ace8d1604475917489a08a39f93f5a28a0"
    assert record["git_identity"]["runtime"] == "c7c940e693719becccef80a918142a27aa050510"
    assert record["git_identity"]["record_commit_is_qualified_source"] is False


def test_r0060_does_not_close_s19_or_authorize_slurm():
    record = _read()
    assert not any(record["scope"].values())
    assert record["next_gate"] == "D-0020 class 2 deployment through preflight-only; class 3 user confirmation before sbatch"


def test_successor_120_sources_resolve_without_external_reads():
    record = _read()
    bindings = {p: v["current_sha256"] for p, v in record["historical_successor_sources"].items()}
    bindings.update(record["introduced_sources"])
    for path, digest in bindings.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, digest)
