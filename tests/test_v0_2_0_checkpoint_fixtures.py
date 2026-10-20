"""Read-only compatibility gates for checkpoints written by PSSolver 0.2.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.v0_2_0_checkpoint_fixtures import replay_case


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "v0_2_0_checkpoints_v1"
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
def test_v020_checkpoint_direct_import_and_bitwise_continuation(
    case, tmp_path
):
    before = _tree(FIXTURES / case)
    replay_case(case, FIXTURES, tmp_path / case)
    assert _tree(FIXTURES / case) == before
