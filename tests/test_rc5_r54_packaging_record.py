"""Static closure record for R5-4 packaging and installed provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_4_packaging_surface.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _r57_successor(relative: str, historical: str) -> str:
    registration = json.loads(R5_7_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if (
        registration is not None
        and registration["historical_sha256"] == historical
    ):
        return registration["current_sha256"]
    return historical


def test_r54_record_binds_candidate_and_exact_sources():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_R5_4_PACKAGING_SURFACE_AND_INSTALLED_PLANE_CPU_CLOSURE"
    )
    assert record["git_identity"] == {
        "branch": "fix/v0.2.0rc5",
        "implementation_commit": "e17209d8384b8be1259aaf91025529bfd4c9e339",
        "location_neutrality_correction_commit": (
            "4678be30785529d39427986c1a6d0819545729e6"
        ),
        "qualified_source_commit": "4678be30785529d39427986c1a6d0819545729e6",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _r57_successor(relative, expected)


def test_r54_contract_keeps_only_the_supported_distribution_surface():
    contract = _record()["contract"]
    assert contract["wheel_top_level"] == [
        "Plane_beris_edwards_stokes",
        "pssolver",
    ]
    assert contract["benchmarks_installed_file_count"] == 0
    assert contract["compatibility_shim_retained"] is True
    assert contract["compatibility_shim_is_numerical_provenance"] is False
    assert contract["implementation_provenance_prefix"] == "pssolver/"
    assert contract["public_plane_without_shim"] == [
        "legacy_production",
        "compiled_v2",
    ]


def test_r54_successor_registry_matches_every_changed_historical_source():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "setup.py",
        "pssolver/configuration/plane_beris_edwards.py",
        "tests/test_fig4_numerics_cli.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == _r57_successor(
            relative,
            registration["current_sha256"],
        )
        assert registration["reason"]


def test_r54_installed_and_source_cpu_closures_are_exact():
    validation = _record()["validation"]
    assert validation["source_full_cpu"] == (
        "3470 passed, 7 skipped, 8 subtests passed, 0 failed"
    )
    assert validation["installed_wheel_targeted_cpu"] == "35 passed, 0 failed"
    assert validation["installed_wheel_applicable_full_cpu"] == (
        "3463 passed, 7 skipped, 8 subtests passed, 0 failed"
    )
    assert validation["direct_wheel_content"] == "PASS"
    assert validation["sdist_rebuilt_wheel_content"] == "PASS"
    assert validation["installed_public_plane_without_shim"] == "2/2 PASS"
    assert validation["source_shadow_import"] is False
    assert validation["pip_check"] == "PASS"


def test_r54_authorizes_only_the_next_local_must_item():
    record = _record()
    assert record["scope"]["H100_required"] is False
    assert record["scope"]["H100_executed"] is False
    assert record["scope"]["PSSolver_Control_modified"] is False
    assert record["scope"]["nematics3d_modified"] is False
    assert record["authorization"] == {
        "r5_4_complete": True,
        "eligible_for_r5_5": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "eligible_for_rc5_release": False,
        "production_default_changed": False,
    }
