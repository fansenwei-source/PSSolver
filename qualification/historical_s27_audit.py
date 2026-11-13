"""Execute frozen stage audits in their own S-26 Git checkout (CPU only).

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


SOURCE_COMMIT = "327a192ba879da0991110eca99f4b69b9a45c84e"
MANIFEST = Path(__file__).with_name("historical_s27_tests.json")


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
    distribution = work / "dist"
    environment.update(PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    commands = [
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distribution),
        ],
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "venv",
            "--system-site-packages",
            str(work / "installed"),
        ],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=checkout, env=environment, capture_output=True)
        (work / f"setup_{index}.stdout").write_bytes(result.stdout)
        (work / f"setup_{index}.stderr").write_bytes(result.stderr)
        assert result.returncode == 0, result.stderr.decode()
    wheels = list(distribution.glob("*.whl"))
    assert len(wheels) == 1
    python = work / "installed/bin/python"
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheels[0]),
        ],
        cwd=work,
        env=environment,
        capture_output=True,
    )
    (work / "install.stdout").write_bytes(result.stdout)
    (work / "install.stderr").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode()
    environment["PSSOLVER_INSTALLED_PYTHON"] = str(python)
    command = [
        str(python),
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
    assert Path(report["pssolver_file"]).is_relative_to(work / "installed")
    assert _git(checkout, "rev-parse", "HEAD") == SOURCE_COMMIT
    return report


def _child(checkout: Path, output: Path, files: list[str], *, collect: bool) -> int:
    # Resolve the Provider from the installed S-26 wheel before adding test sources.
    import pssolver

    sys.path.insert(0, str(checkout))
    # The historical checkout can shadow the current qualification package.
    order_path = Path(__file__).with_name("historical_successor_order.py")
    spec = importlib.util.spec_from_file_location("_historical_successor_order", order_path)
    assert spec is not None and spec.loader is not None
    order = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order)
    order.pin(checkout)
    import pytest
    import torch

    # Reuse the S-25 trial launch's frozen CPU settings, before any test executes.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    assert not Path(pssolver.__file__).resolve().is_relative_to(checkout.resolve())
    report = {
        "pssolver_file": pssolver.__file__,
        "nodeids": [],
        "outcomes": {},
        "deselected": [],
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }

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
