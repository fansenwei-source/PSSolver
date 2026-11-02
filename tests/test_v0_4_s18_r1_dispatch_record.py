"""Frozen R1 dispatch scope and exact forward bindings, not live branch state."""
import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_112_r1_dispatch.json"


def _record():
    return json.loads(RECORD.read_text())


def test_r1_dispatch_has_a_proven_shadow_predecessor_and_no_tolerance():
    record = _record()
    assert record["production_dispatch_switched"] is True
    assert record["decision"]["id"] == "D-0018"
    assert record["nonzero_tolerance_authorized"] is False
    assert record["review_complete"] is False
    assert record["legacy_negative_control"]["production_fallback"] is False
    assert not any(record["external_actions_authorized"].values())


def test_r1_dispatch_binds_only_exact_forward_source_changes():
    record = _record()
    for path, entry in record["historical_successor_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, entry["current_sha256"])
    for path, digest in record["introduced_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, digest)


def test_r1_concrete_and_consumer_boundary_blobs_were_frozen_at_switch():
    record = _record()
    before, after = (record["git_identity"][key] for key in ("batch_parent", "dispatch_source"))
    for path in record["frozen_sources"]:
        assert subprocess.check_output(["git", "rev-parse", f"{before}:{path}"], cwd=ROOT) == subprocess.check_output(["git", "rev-parse", f"{after}:{path}"], cwd=ROOT)
    changed = subprocess.check_output(["git", "diff", "--name-only", before, after, "pssolver/"], cwd=ROOT, text=True).splitlines()
    assert sorted(changed) == sorted(record["runtime_delta_paths"])


def test_r1_intentionally_keeps_generic_declaration_and_metadata_planner():
    record = _record()
    assert set(record["intentional_no_change"]) == {"pssolver/configuration/simulation.py", "pssolver/configuration/package_construction.py"}
    assert record["real_nfs_triggered"] is False
    assert record["consumer_migration_triggered"] is False
    assert record["gpu_specific_paths_changed"] is False
