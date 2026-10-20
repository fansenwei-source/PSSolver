"""Read-only compatibility gates for checkpoints written by PSSolver 0.2.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import benchmarks.v0_2_0_checkpoint_fixtures as compatibility
from benchmarks.v0_2_0_checkpoint_fixtures import replay_case


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "v0_2_0_checkpoints_v1"
ORACLE = (
    ROOT / "tests" / "oracles" / "v0_2_0_checkpoint_continuations_v1"
)
CASES = (
    "functional_periodic",
    "functional_channel",
    "production_periodic",
    "production_channel_complete_stress",
    "production_plane_legacy",
    "production_legacy_channel",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tree_sha256(directory: Path) -> str:
    encoded = json.dumps(
        _tree(directory), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def test_v020_fixture_manifest_binds_the_published_wheel_and_release():
    manifest = _manifest()
    assert manifest["fixture_schema_version"] == 1
    assert manifest["release"] == {
        "commit": "6d2934e73b543f6652d112733657775333e0f396",
        "pssolver_import": manifest["release"]["pssolver_import"],
        "published_wheel_sha256": (
            "ab4beef1a347568e8ad4162291feb62e94e179849853ca1417583634649a1151"
        ),
        "source_shadow_import": False,
        "tag": "v0.2.0",
        "version": "0.2.0",
    }
    assert "/site-packages/pssolver/__init__.py" in (
        manifest["release"]["pssolver_import"]
    )
    assert manifest["policy"] == {
        "read_only_fixture": True,
        "direct_import_required": True,
        "continuation_equivalence": "bitwise",
        "continuation_steps": 1,
        "fixture_mutation_allowed": False,
    }
    assert set(manifest["cases"]) == set(CASES)


@pytest.mark.parametrize("case", CASES)
def test_v020_checkpoint_tree_is_provenance_bound(case):
    record = _manifest()["cases"][case]
    directory = FIXTURES / case
    assert _tree(directory) == record["files"]
    assert len(record["files"]) == record["file_count"]
    assert _tree_sha256(directory) == record["tree_sha256"]
    assert record["completed_steps"] == 1
    assert record["continuation_steps"] == 1


@pytest.mark.parametrize("case", CASES)
def test_v020_checkpoint_direct_import_and_build_aware_continuation(
    case, tmp_path
):
    before = _tree(FIXTURES / case)
    report = replay_case(case, FIXTURES, tmp_path / case, ORACLE)
    assert report["payload"]["equivalence"] == "bitwise"
    assert report["payload"]["payload_count"] > 0
    assert report["continuation"]["mode"] in {
        "bitwise",
        "dtype_epsilon",
    }
    assert all(
        item["passed"]
        for item in report["continuation"]["arrays"].values()
    )
    assert _tree(FIXTURES / case) == before


def test_v020_continuation_oracle_is_separate_and_provenance_bound():
    manifest = json.loads(
        (ORACLE / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["oracle_schema_version"] == 1
    assert manifest["checkpoint_fixture_tree_sha256"] == _tree_sha256(FIXTURES)
    assert manifest["tolerance_policy"] == {
        "matching_build": "bitwise",
        "other_builds": "dtype_epsilon",
        "epsilon_multiplier": 8.0,
    }
    assert set(manifest["cases"]) == set(CASES)
    for case in CASES:
        for record in manifest["cases"][case]["arrays"].values():
            path = ORACLE / record["file"]
            assert _sha256(path) == record["file_sha256"]


def _synthetic_oracle(tmp_path, reference, fingerprint):
    directory = tmp_path / "oracle" / "case"
    directory.mkdir(parents=True)
    path = directory / "value.npy"
    np.save(path, reference, allow_pickle=False)
    manifest = {
        "numerical_build_fingerprint": fingerprint,
        "cases": {
            "case": {
                "arrays": {
                    "value": {
                        "file": "case/value.npy",
                        "file_sha256": _sha256(path),
                    }
                }
            }
        },
    }
    (directory.parent / "manifest.json").write_text(json.dumps(manifest))
    return directory.parent


def test_build_aware_continuation_requires_bits_for_matching_build(
    tmp_path, monkeypatch
):
    fingerprint = {"build": "reference"}
    reference = np.array([1.0], dtype=np.float64)
    oracle = _synthetic_oracle(tmp_path, reference, fingerprint)
    actual = np.nextafter(reference, np.inf)
    record = compatibility._array_record(actual)
    monkeypatch.setattr(
        compatibility, "numerical_build_fingerprint", lambda: fingerprint
    )
    with pytest.raises(AssertionError, match="continued checkpoint state"):
        compatibility._compare_continuation(
            {"value": record},
            {"value": actual},
            {"value": compatibility._array_record(reference)},
            oracle,
            "case",
        )


def test_build_aware_continuation_uses_dtype_epsilon_for_other_build(
    tmp_path, monkeypatch
):
    reference_fingerprint = {"build": "reference"}
    observed_fingerprint = {"build": "other"}
    reference = np.array([1.0], dtype=np.float32)
    oracle = _synthetic_oracle(tmp_path, reference, reference_fingerprint)
    epsilon = np.finfo(np.float32).eps
    actual = reference + np.float32(2.0 * epsilon)
    record = compatibility._array_record(actual)
    monkeypatch.setattr(
        compatibility,
        "numerical_build_fingerprint",
        lambda: observed_fingerprint,
    )
    report = compatibility._compare_continuation(
        {"value": record},
        {"value": actual},
        {"value": compatibility._array_record(reference)},
        oracle,
        "case",
    )
    assert report["mode"] == "dtype_epsilon"
    assert report["arrays"]["value"]["atol"] == 8.0 * epsilon

    outside_tolerance = reference + np.float32(32.0 * epsilon)
    with pytest.raises(AssertionError, match="continued checkpoint state"):
        compatibility._compare_continuation(
            {"value": compatibility._array_record(outside_tolerance)},
            {"value": outside_tolerance},
            {"value": compatibility._array_record(reference)},
            oracle,
            "case",
        )
