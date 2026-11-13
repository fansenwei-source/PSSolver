"""Execute frozen stage audits in their own v0.4.0 Git checkout (CPU only).

This runner never puts historical code into the current Provider interpreter.
The single parent test checks the exact child nodeid set and every outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_COMMIT = "4c12228aba01e8483e80d28a1471ab72384d5fda"
MANIFEST = Path(__file__).with_name("historical_stage_tests.json")


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError("PSSOLVER_TEST_PREREQUISITE: historical Git checkout required") from exc
    if result.returncode:
        raise RuntimeError(
            "PSSOLVER_TEST_PREREQUISITE: historical Git checkout required\n" + result.stderr
        )
    return result.stdout.strip()


def run_audit(repository: Path, work: Path, *, manifest_path: Path = MANIFEST) -> dict:
    """Check frozen sources and run the exact manifest without skipped audits."""
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_commit"] == SOURCE_COMMIT
    _git(repository, "cat-file", "-e", SOURCE_COMMIT + "^{commit}")
    work.mkdir(parents=True, exist_ok=False)
    checkout = work / "source"
    _git(repository, "clone", "--shared", "--no-checkout", str(repository), str(checkout))
    _git(checkout, "checkout", "--detach", SOURCE_COMMIT)
    for item in manifest["files"]:
        assert _git(checkout, "rev-parse", SOURCE_COMMIT + ":" + item["path"]) == item["blob"]
        assert hashlib.sha256((checkout / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    result_path = work / "result.json"
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "--child",
        str(checkout),
        str(result_path),
        *manifest.get("selected_nodeids", [item["path"] for item in manifest["files"]]),
    ]
    with (work / "stdout.txt").open("w") as out, (work / "stderr.txt").open("w") as err:
        completed = subprocess.run(
            command, cwd=checkout, env=environment, stdin=subprocess.DEVNULL, stdout=out, stderr=err
        )
    assert completed.returncode == 0, (work / "stdout.txt").read_text() + (
        work / "stderr.txt"
    ).read_text()
    report = json.loads(result_path.read_text())
    assert report["nodeids"] == manifest["nodeids"]
    assert report["deselected"] == []
    assert report["outcomes"] == {nodeid: "passed" for nodeid in manifest["nodeids"]}
    assert Path(report["pssolver_file"]).is_relative_to(checkout)
    assert _git(checkout, "rev-parse", "HEAD") == SOURCE_COMMIT
    return report


def _child(checkout: Path, output: Path, files: list[str], *, collect: bool) -> int:
    sys.path.insert(0, str(checkout))
    # The historical checkout can shadow the current qualification package.
    order_path = Path(__file__).with_name("historical_successor_order.py")
    spec = importlib.util.spec_from_file_location("_historical_successor_order", order_path)
    assert spec is not None and spec.loader is not None
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    order.pin(checkout)
    import pytest

    import pssolver

    assert Path(pssolver.__file__).resolve().is_relative_to(checkout.resolve())
    report = {"pssolver_file": pssolver.__file__, "nodeids": [], "outcomes": {}, "deselected": []}

    class Recorder:
        def __init__(self, data):
            self.data = data

        def pytest_collection_finish(self, session):
            report["nodeids"] = sorted(item.nodeid for item in session.items)

        def pytest_deselected(self, items):
            report["deselected"].extend(item.nodeid for item in items)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.outcome != "passed":
                self.data["outcomes"][report.nodeid] = report.outcome

    recorder = Recorder(report)
    arguments = [
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml=" + str(output.with_suffix(".xml")),
        *files,
    ]
    if collect:
        arguments.append("--collect-only")
    code = pytest.main(arguments, plugins=[recorder])
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    raise SystemExit(_child(args.checkout, args.output, args.files, collect=args.collect))
