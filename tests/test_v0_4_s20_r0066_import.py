"""Forward-only R-0066 import; no external archive reads in pytest."""
import hashlib
import json
from pathlib import Path
import subprocess
from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_131_r0066_import.json"

def test_r0066_is_only_the_closeout_interim_gate():
    r = json.loads(RECORD.read_text())
    assert r["classification"] == "INTERIM_PASS_S20_CLOSEOUT_IMPORTED"
    assert r["review"]["classification"] == "INTERIM_PASS_S20_CLOSEOUT"
    assert r["review"]["registry_updates"] == []
    assert r["review"]["new_findings"] == []
    assert r["final_s20_pass"] is False
    assert r["record_only_triage_is_not_a_code_fix"] is True

def test_r0066_binds_the_reviewed_target_not_a_future_head():
    r = json.loads(RECORD.read_text())
    target = r["git_identity"]["reviewed_target"]
    assert target == "1ec1a8bac210d90be1b4c3b41020401a46f0a903"
    subprocess.run(["git", "merge-base", "--is-ancestor", target, "HEAD"], cwd=ROOT, check=True)
    actual = subprocess.check_output(["git", "rev-parse", target+":pssolver"], cwd=ROOT, text=True).strip()
    assert actual == r["qualified_runtime_tree"]
    assert r["archive"]["checksums_entries"] == 40
    assert r["archive"]["checksums_sha256"] == "5bc7c5bf2b57c39578c1a6629981677ad6a591247ca509320a33332d5a044a3d"

def test_r0066_import_sources_are_forward_bound():
    r = json.loads(RECORD.read_text())
    for path, digest in r["introduced_sources"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == resolve_s03_successor(path,digest)
    assert r["historical_successor_sources"] == {}
    assert r["registry_changed"] is False
    assert r["runtime_changed"] is False
