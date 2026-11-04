"""Version-only v0.4.0 freeze; archived snapshots may advance explicitly."""
import hashlib
import json
from pathlib import Path
import subprocess
from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_4_0.json"
SUCCESSOR = ROOT / "notes/PSSolver_v0_3_0_s03_successor_132_v040_freeze.json"

def test_v040_freeze_preserves_the_s19_runtime_except_version():
    r=json.loads(RECORD.read_text())
    assert r["source"]["version"]=="0.4.0"
    assert r["source"]["runtime_delta_allowlist"]==["pssolver/_version.py"]
    for rev in (r["source"]["parent"],r["source"]["qualified_runtime"]):
        tree=subprocess.check_output(["git","rev-parse",rev+":pssolver"],cwd=ROOT,text=True).strip()
        assert tree==r["source"]["qualified_runtime_tree"]
    assert r["source"]["tag_target_policy"]=="exact independently accepted candidate commit"

def test_v040_gates_distinguish_candidate_and_final_acceptance():
    r=json.loads(RECORD.read_text())
    assert list(r["completion_gates"])==[f"G-{i:02d}" for i in range(1,10)]
    assert r["completion_gates"]["G-09"]=="candidate review, publication, and post-release acceptance pending"
    assert r["known_limits"]["V-63"]=="accepted, not fixed (D-0024)"
    assert r["known_limits"]["R1-R9"]=="remaining scope deferred after 0.4 (D-0024)"
    assert r["known_limits"]["D-0013"]=="historical 0.3 exception does not extend to 0.4"
    assert r["record_only_closeout_claimed_as_code_fixes"] is False
    assert r["registry_snapshot"]["status_counts"]=={"verified":65,"accepted":3,"deferred":2}

def test_v040_release_is_not_authorized_by_local_freeze():
    r=json.loads(RECORD.read_text())
    assert r["release"]["asset_names"]==["pssolver-0.4.0-py3-none-any.whl","pssolver-0.4.0.tar.gz"]
    assert r["release"]["verify_tag"] is True
    assert r["release"]["prerelease"] is False
    assert r["release"]["latest"] is True
    assert not any(r["external_actions_authorized"].values())
    body=(ROOT/"notes/pssolver_v0_4_0_release_notes.txt").read_text()
    assert "/home/" not in body and "PSSolver-Control" not in body
    assert "nonfinite backward gradients" in body
    assert "does not extend to v0.4.0" in body

def test_v040_changed_sources_are_forward_bound():
    s=json.loads(SUCCESSOR.read_text())
    for path, item in s["historical_successor_sources"].items():
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        assert actual==resolve_s03_successor(path,item["historical_sha256"])
    for path, digest in s["introduced_sources"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==resolve_s03_successor(path,digest)
    assert s["historical_records_rewritten"] is False
