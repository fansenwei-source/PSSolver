"""Unit tests for the versioned RC4.2.5 installed-wheel runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_rc425_installed_checkpoint_closure.py"
PLAN = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc425_installed_cpu_qualification_plan.json"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "run_rc425_installed_checkpoint_closure",
        RUNNER,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_rc425_runner_is_importable_without_consumer_package():
    module = _module()
    assert module.CLASSIFICATION == (
        "PASS_RC4_2_5_INSTALLED_WHEEL_CPU_CLOSURE"
    )
    assert callable(module.main)


def test_rc425_runner_parser_requires_both_installed_test_groups():
    parser = _module()._parser()
    arguments = parser.parse_args(
        [
            "--provider-purelib",
            "/provider",
            "--consumer-purelib",
            "/consumer",
            "--expected-provider-tests",
            "35",
            "--expected-consumer-tests",
            "60",
            "--output",
            "/result.json",
        ]
    )
    assert arguments.expected_provider_tests == 35
    assert arguments.expected_consumer_tests == 60
    assert arguments.provider_test == []
    assert arguments.consumer_test == []


def test_rc425_runner_atomic_json_round_trip_and_existing_target_guard(tmp_path):
    module = _module()
    output = tmp_path / "result.json"
    value = {"finite": True, "count": 95}
    module._atomic_json(output, value)
    assert json.loads(output.read_text()) == value
    assert not list(tmp_path.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError, match="already exists"):
        module._atomic_json(output, value)


def test_rc425_runner_collector_rejects_skip_xfail_and_setup_failure():
    collector = _module()._PytestResults()
    collector.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_ok",
            when="call",
            passed=True,
            failed=False,
            skipped=False,
        )
    )
    collector.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_skip",
            when="setup",
            passed=False,
            failed=False,
            skipped=True,
            longrepr="reason",
        )
    )
    collector.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_xfail",
            when="call",
            passed=False,
            failed=False,
            skipped=True,
            wasxfail="expected",
        )
    )
    metadata = collector.to_metadata()
    assert metadata["passed"] == 1
    assert len(metadata["skipped"]) == 1
    assert metadata["xfailed"] == ["test_xfail"]


def test_rc425_runner_path_and_file_guards(tmp_path):
    module = _module()
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = directory / "test.py"
    regular.write_text("pass\n")
    link = directory / "link.py"
    link.symlink_to(regular)

    assert module._ordinary_directory(str(directory), "directory") == directory
    assert module._ordinary_file(str(regular), "test") == regular
    assert module._is_relative_to(regular, directory) is True
    with pytest.raises(ValueError, match="ordinary file"):
        module._ordinary_file(str(link), "test")


def test_rc425_plan_freezes_installed_provider_and_consumer_matrix():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "ready_for_execution"
    assert plan["source_identity"]["provider_parent"] == (
        "573daaf1e2d4ad7e54c6c084a6fbb04126fd4966"
    )
    assert plan["source_identity"]["consumer_source_commit"] == (
        "03dbc39975978d9d0b7f84965015da24f9d3c206"
    )
    assert plan["provider_tests"]["expected"] == 35
    assert plan["consumer_tests"]["expected"] == 60
    assert plan["runner"]["expected_total_tests"] == 95
    assert plan["runner"]["skip_allowed"] == 0
    assert plan["runner"]["xfail_allowed"] == 0
    assert plan["scope"]["H100_authorized"] is False
    assert plan["scope"]["PSSolver_Control_source_change_authorized"] is False
