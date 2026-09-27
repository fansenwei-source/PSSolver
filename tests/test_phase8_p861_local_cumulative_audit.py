"""P8.6.1 local cumulative evidence, catalog, package, and rejection audit."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "benchmarks/analyze_phase8_cumulative_closure.py"


def _load_analyzer():
    specification = importlib.util.spec_from_file_location(
        "analyze_phase8_cumulative_closure",
        ANALYZER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load_analyzer()


@pytest.fixture(scope="module")
def installed_probe(analyzer):
    return analyzer.build_installed_wheel_probe(ROOT)


@pytest.fixture(scope="module")
def report(analyzer, installed_probe):
    return analyzer.analyze(ROOT, installed_wheel_probe=installed_probe)


def test_p861_fail_closed_analyzer_binds_all_authoritative_prerequisites(report):
    assert report["classification"] == "PASS_P8_6_1_LOCAL_CUMULATIVE_AUDIT"
    assert [item["phase"] for item in report["prerequisites"]] == [
        "P8.1",
        "P8.2",
        "P8.3",
        "P8.4.5",
        "P8.5.6",
    ]
    assert {item["status"] for item in report["prerequisites"]} == {"pass"}


def test_p861_live_and_installed_catalogs_are_exactly_equal(report):
    source = report["source_catalog"]
    installed = report["installed_wheel"]
    assert source["catalog"] == installed["catalog"]
    assert installed["source_shadow_import"] is False
    assert installed["wheel_filename"].endswith(".whl")
    assert len(installed["wheel_sha256"]) == 64
    assert installed["wheel_member_count"] > 0


def test_p861_freezes_six_runtime_paths_and_non_promoted_robin(report):
    catalog = report["source_catalog"]["catalog"]
    runtime_paths = [
        path
        for combination in catalog["qualified_combinations"]
        for path in combination["runtime_paths"]
    ]
    assert runtime_paths == [
        "legacy_production",
        "compiled_v2",
        "legacy_channel",
        "compiled_channel_v2",
        "periodic_spectral",
        "channel_complete_stress",
    ]
    robin = next(
        value for value in catalog["boundary_policies"] if value["key"] == "robin"
    )
    assert robin["declarable"] is True
    assert robin["executable"] is False
    assert robin["qualified_applications"] == []


def test_p861_structured_rejection_matrix_has_executable_test_coverage(report):
    matrix = report["rejection_matrix"]
    assert len(matrix) == 8
    assert [item["case"] for item in matrix] == [
        "unregistered_model_geometry",
        "public_complete_timestep_robin_q",
        "prescribed_q_on_unqualified_geometry",
        "wall_policy_on_periodic_face",
        "geometry_incompatible_velocity_or_pressure_policy",
        "nonzero_prescribed_neumann_flux",
        "dynamic_spatial_callable_or_trainable_robin_data",
        "checkpoint_identity_or_payload_mismatch",
    ]
    assert all(item["status"] == "pass" for item in matrix)
    assert all(item["evidence"] == "executable_repository_test" for item in matrix)
    assert all(len(item["source_sha256"]) == 64 for item in matrix)


def test_p861_defaults_and_scope_remain_frozen(report):
    assert report["source_catalog"]["defaults"] == {
        "plane": "legacy_production",
        "channel": "legacy_channel",
    }
    assert not any(report["scope"].values())
    assert report["authorization"] == {
        "p8_6_1_complete": True,
        "eligible_for_p8_6_2_h100_planning": True,
        "p8_6_2_h100_executed": False,
        "phase_9_authorized": False,
    }


def test_p861_analyzer_rejects_installed_catalog_drift(analyzer, installed_probe):
    changed = copy.deepcopy(installed_probe)
    changed["catalog"]["models"][0]["key"] = "silently_changed"
    with pytest.raises(RuntimeError, match="public model catalog changed"):
        analyzer.analyze(ROOT, installed_wheel_probe=changed)


def test_p861_report_is_strict_json(report):
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["classification"] == "PASS_P8_6_1_LOCAL_CUMULATIVE_AUDIT"
