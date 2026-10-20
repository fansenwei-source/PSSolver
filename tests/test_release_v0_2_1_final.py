"""Formal GitHub release record for PSSolver v0.2.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "pssolver_v0_2_1_release.json"
PREPARATION = ROOT / "notes" / "pssolver_v0_2_1_package_preparation.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v021_release_record_binds_the_exact_source_and_annotated_tag():
    record = _read(RECORD)
    assert record["status"] == "released"
    assert record["classification"] == "PASS_V0_2_1_FORMAL_GITHUB_RELEASE"
    assert record["source"]["release_source_commit"] == (
        "ca9c6729670f07639e7877881c777d1653f4a592"
    )
    assert record["source"]["record_commit_is_release_source"] is False
    assert record["publication"]["annotated_tag"] == "v0.2.1"
    assert record["publication"]["tag_object"] == (
        "daf1270b1f046ac767045cbbe536fd185599d83f"
    )
    assert record["publication"]["tag_target"] == record["source"][
        "release_source_commit"
    ]


def test_v021_release_record_binds_github_release_and_assets():
    publication = _read(RECORD)["publication"]
    assert publication["github_release_id"] == 418261343
    assert publication["draft"] is False
    assert publication["prerelease"] is False
    assert publication["assets"] == {
        "pssolver-0.2.1-py3-none-any.whl": {
            "asset_id": 658969850,
            "size_bytes": 801264,
            "sha256": "0f0f1c43a920582d3d707d285fe06502ff1852e6cca24f37872b06669201da8a",
        },
        "pssolver-0.2.1.tar.gz": {
            "asset_id": 658969846,
            "size_bytes": 1399709,
            "sha256": "cf795a976494dd552f3db843ba88185f77a0b9510ef3e60d4a9b9582413b6eba",
        },
    }


def test_v021_release_uses_the_frozen_package_preparation_evidence():
    record = _read(RECORD)
    qualification = record["qualification"]
    assert _sha256(PREPARATION) == qualification[
        "package_preparation_record_sha256"
    ]
    preparation = _read(PREPARATION)
    assert preparation["source"]["release_source_commit"] == record["source"][
        "release_source_commit"
    ]
    assert preparation["build"]["wheel"]["sha256"] == record["publication"][
        "assets"
    ]["pssolver-0.2.1-py3-none-any.whl"]["sha256"]
    assert preparation["build"]["sdist"]["sha256"] == record["publication"][
        "assets"
    ]["pssolver-0.2.1.tar.gz"]["sha256"]


def test_v021_release_preserves_qualified_runtime_boundaries():
    source = _read(RECORD)["source"]
    for key in (
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "provider_identity_changed",
        "production_default_changed",
    ):
        assert source[key] is False
    assert source["runtime_delta_from_h100_qualified_source"] == [
        "pssolver/_version.py: 0.2.0 -> 0.2.1"
    ]


def test_v021_release_records_qualification_without_new_h100_execution():
    qualification = _read(RECORD)["qualification"]
    assert qualification["cpu_suite"] == {
        "passed": 4134,
        "failed": 0,
        "skipped": 1,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    assert qualification["installed_wheel_n1_checkpoint_tests"] == {
        "passed": 13,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert qualification["h100_job_id"] == 10902665
    assert qualification["h100_rerun_required"] is False
    assert qualification["nematics3d_modified"] is False


def test_v021_release_excludes_pypi_and_additional_h100_actions():
    record = _read(RECORD)
    assert record["publication"]["pypi_published"] is False
    assert record["publication"]["h100_rerun_performed"] is False
    assert record["scope"] == {
        "release_tag_rewritten": False,
        "release_assets_replaced": False,
        "pypi_modified": False,
        "additional_h100_job_submitted": False,
        "record_is_evidence_only": True,
    }
    assert record["authorization"]["pypi_publication_authorized"] is False
    assert record["authorization"]["additional_h100_authorized"] is False
