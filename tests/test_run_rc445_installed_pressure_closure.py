"""Unit tests for the RC4.4.5 installed-wheel pressure runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks/run_rc445_installed_pressure_closure.py"
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc445_installed_cpu_qualification_plan.json"


def _module():
    specification = importlib.util.spec_from_file_location(
        "run_rc445_installed_pressure_closure",
        RUNNER,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_runner_imports_without_pssolver_side_effects():
    module = _module()
    assert module.CLASSIFICATION == (
        "PASS_RC4_4_5_INSTALLED_WHEEL_PRESSURE_CPU_CLOSURE"
    )
    assert callable(module.main)


def test_atomic_json_round_trip_and_existing_target_guard(tmp_path):
    module = _module()
    output = tmp_path / "result.json"
    value = {"passed": True, "tests": 85}
    module._atomic_json(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert not list(tmp_path.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError, match="already exists"):
        module._atomic_json(output, value)


def test_collector_exposes_fail_closed_outcomes():
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
    metadata = collector.to_metadata()
    assert metadata["passed"] == 1
    assert metadata["skipped"][0]["nodeid"] == "test_skip"


def test_path_guards_reject_symlinked_test_input(tmp_path):
    module = _module()
    directory = tmp_path / "directory"
    directory.mkdir()
    regular = directory / "test.py"
    regular.write_text("pass\n", encoding="utf-8")
    link = directory / "link.py"
    link.symlink_to(regular)
    assert module._ordinary_directory(str(directory), "directory") == directory
    assert module._ordinary_file(str(regular), "test") == regular
    with pytest.raises(ValueError, match="ordinary file"):
        module._ordinary_file(str(link), "test")


def test_plan_freezes_the_provider_only_cpu_boundary():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "ready_for_execution"
    assert plan["source_identity"]["runtime_source_commit"] == (
        "dc8c87abc7b216231d09ca12fcc7e76b2249a7ab"
    )
    assert plan["source_gate"]["expected_tests"] == 85
    assert plan["installed_wheel_gate"]["expected_tests"] == 85
    assert plan["pressure_closure"]["issues"] == ["B25", "B28", "B29"]
    assert plan["qualification_boundary"]["H100_authorized"] is False
    assert plan["qualification_boundary"]["PSSolver_Control_change_authorized"] is False
    assert plan["qualification_boundary"]["nematics3d_change_authorized"] is False
