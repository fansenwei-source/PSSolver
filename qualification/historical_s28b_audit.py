"""Run the S-28b historical contracts from the accepted predecessor on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_COMMIT = "49d1c32b66ef0088fed4ea13ab71df4d1a1bf802"
MANIFEST = Path(__file__).with_name("s28b_historical_tests.json")
MODES = ("installed", "source")
SOURCE_ONLY = {"tests/test_dal.py"}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "PSSOLVER_TEST_PREREQUISITE: historical Git checkout required\n" + result.stderr
        )
    return result.stdout.strip()


def _run(command, cwd, environment, log):
    with log.with_suffix(".stdout").open("w") as out, log.with_suffix(".stderr").open("w") as err:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            check=False,
        )
    assert result.returncode == 0, (
        log.with_suffix(".stdout").read_text() + log.with_suffix(".stderr").read_text()
    )


def run_audit(repository: Path, work: Path, *, manifest_path: Path = MANIFEST) -> dict:
    """Verify original bytes, exact node identities, and a passing outcome for every node."""
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["source_commit"] == SOURCE_COMMIT
    _git(repository, "cat-file", "-e", SOURCE_COMMIT + "^{commit}")
    work.mkdir(parents=True, exist_ok=False)
    checkout = work / "source"
    _git(repository, "clone", "--shared", "--no-checkout", str(repository), str(checkout))
    _git(checkout, "checkout", "--detach", SOURCE_COMMIT)
    assert _git(checkout, "rev-parse", "HEAD") == SOURCE_COMMIT
    paths = {node.split("::", 1)[0] for node in manifest["nodeids"]}
    assert paths == {item["path"] for item in manifest["files"]}
    for item in manifest["files"]:
        path = item["path"]
        assert _git(checkout, "rev-parse", SOURCE_COMMIT + ":" + path) == item["blob"]
        assert hashlib.sha256((checkout / path).read_bytes()).hexdigest() == item["sha256"]

    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PIP_NO_CACHE_DIR="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    dist = work / "dist"
    venv = work / "installed"
    _run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist),
        ],
        checkout,
        environment,
        work / "build",
    )
    _run(
        [sys.executable, "-I", "-B", "-m", "venv", "--system-site-packages", str(venv)],
        work,
        environment,
        work / "venv",
    )
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    installed = venv / "bin/python"
    _run(
        [
            str(installed),
            "-I",
            "-B",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheels[0]),
        ],
        work,
        environment,
        work / "install",
    )
    environment["PSSOLVER_INSTALLED_PYTHON"] = str(installed)

    reports = {}
    for mode in MODES:
        nodeids = manifest["modes"][mode]
        assert nodeids and len(nodeids) == len(set(nodeids))
        assert all(
            (node.split("::", 1)[0] in SOURCE_ONLY) == (mode == "source") for node in nodeids
        )
        output = work / f"{mode}.json"
        python = sys.executable if mode == "source" else str(installed)
        _run(
            [
                python,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                mode,
                str(checkout),
                str(output),
                str(manifest_path.resolve()),
            ],
            checkout,
            environment,
            work / f"{mode}_run",
        )
        report = json.loads(output.read_bytes())
        assert report["nodeids"] == nodeids
        assert report["outcomes"] == dict.fromkeys(nodeids, "passed")
        assert report["deselected"] == []
        assert (
            Path(report["pssolver_file"])
            .resolve()
            .is_relative_to((checkout if mode == "source" else venv).resolve())
        )
        reports[mode] = report
    assert sorted(sum((reports[mode]["nodeids"] for mode in MODES), [])) == manifest["nodeids"]
    return reports


def _child(checkout: Path, output: Path, manifest: Path, mode: str) -> int:
    nodeids = json.loads(manifest.read_bytes())["modes"][mode]
    if mode == "source":
        sys.path.insert(0, str(checkout))
    import pssolver

    if mode == "installed":
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

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    evidence = {
        "pssolver_file": pssolver.__file__,
        "argv": sys.argv,
        "nodeids": [],
        "outcomes": {},
        "deselected": [],
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }

    class Recorder:
        def pytest_collection_finish(self, session):
            evidence["nodeids"] = sorted(item.nodeid for item in session.items)

        def pytest_deselected(self, items):
            evidence["deselected"].extend(item.nodeid for item in items)

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.outcome != "passed":
                evidence["outcomes"][report.nodeid] = report.outcome

    code = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--junitxml=" + str(output.with_suffix(".xml")), *nodeids],
        plugins=[Recorder()],
    )
    output.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    return int(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    raise SystemExit(_child(args.checkout, args.output, args.manifest, args.mode))
