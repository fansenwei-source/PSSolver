import json

from benchmarks.run_v0_3_s09_installed_consumer_cpu import (
    ALLOWED_SKIPS,
    Results,
    _atomic_json,
)


def test_installed_consumer_cpu_contract_has_only_the_six_existing_skips():
    assert len(ALLOWED_SKIPS) == 6
    assert "tests/test_warmup_experiments.py::test_r5_benchmark" in ALLOWED_SKIPS
    assert sum(value.startswith("tests/test_gpu_optional.py::") for value in ALLOWED_SKIPS) == 3
    assert sum(value.startswith("tests/test_checkpoint_memory.py::") for value in ALLOWED_SKIPS) == 2


def test_installed_consumer_result_collector_starts_empty():
    assert Results().metadata() == {
        "passed": [],
        "failed": [],
        "skipped": [],
        "xfailed": [],
        "xpassed": [],
        "collection_errors": [],
    }


def test_installed_consumer_result_collector_records_setup_phase_skips():
    class Report:
        when = "setup"
        failed = False
        skipped = True
        nodeid = "tests/test_gpu_optional.py::test_cuda"
        longrepr = "CUDA unavailable"

    results = Results()
    results.pytest_runtest_logreport(Report())
    assert results.metadata()["skipped"] == [
        {"nodeid": Report.nodeid, "reason": Report.longrepr}
    ]


def test_installed_consumer_result_is_persisted_before_a_nonzero_exit(tmp_path, monkeypatch):
    from benchmarks import run_v0_3_s09_installed_consumer_cpu as runner

    monkeypatch.setattr(runner, "execute", lambda _args: {"passed": False, "why": "gate"})
    output = tmp_path / "result.json"
    assert runner.main([
        "--support-root", str(tmp_path),
        "--provider-reference", str(tmp_path),
        "--junit", str(tmp_path / "junit.xml"),
        "--output", str(output),
    ]) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passed": False,
        "why": "gate",
    }


def test_installed_consumer_atomic_json_refuses_to_overwrite(tmp_path):
    output = tmp_path / "result.json"
    _atomic_json(output, {"passed": True})
    try:
        _atomic_json(output, {"passed": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing installed-consumer result was overwritten")
