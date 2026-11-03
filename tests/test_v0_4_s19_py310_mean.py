"""S-19 test-only Python 3.10 repair and its frozen forward evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_114_s19_py310_mean.json"


def test_s19_mean_fix_is_only_a_synthetic_test_change():
    record = json.loads(RECORD.read_text())
    before = record["git_identity"]["implementation_parent"]
    after = record["git_identity"]["implementation_source"]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", before, after], cwd=ROOT, text=True
    ).splitlines()
    assert changed == ["tests/test_analyze_v0_3_s09_h100.py"]
    assert record["runtime_changed"] is False
    assert record["frozen_analyzer_changed"] is False
    assert record["thresholds_changed"] is False
    assert record["skip_xfail_added"] is False


def test_s19_synthetic_profiles_use_fmean_exactly(tmp_path):
    from tests.test_analyze_v0_3_s09_h100 import _write_profiles

    profile_root = tmp_path / "profiles"
    _write_profiles(profile_root)
    paths = list(profile_root.glob("*.json"))
    assert len(paths) == 192
    for path in paths:
        profile = json.loads(path.read_text())
        timing = profile["timing"]
        assert timing["mean_timestep_seconds"] == statistics.fmean(
            timing["samples_seconds"]
        )


def test_s19_mean_record_retains_the_exact_baseline_negative_control():
    record = json.loads(RECORD.read_text())
    assert len(record["baseline_expected_failures"]) == 3
    assert record["baseline_error_message"] == (
        "baseline_plane_legacy_production_plane_R128_trial1 mean is not raw-sample mean"
    )
    assert record["original_baseline_must_remain_unmodified"] is True
    assert record["overlay_must_pass_without_additional_exemptions"] is True
    assert record["issue_snapshot"]["status"] == "claimed_fixed"
    assert record["issue_snapshot"]["id"] == "V-62"


def test_s19_mean_forward_bindings_are_exact():
    record = json.loads(RECORD.read_text())
    sources = {path: entry["current_sha256"] for path, entry in record["historical_successor_sources"].items()}
    sources.update(record["introduced_sources"])
    for path, digest in sources.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, digest)
