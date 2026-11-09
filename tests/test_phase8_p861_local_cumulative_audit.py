"""P8.6.1 local cumulative evidence, catalog, package, and rejection audit."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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


def _analyze_source(installed_probe):
    """Keep the frozen source-side audit in a genuine source interpreter.

    The parent suite may intentionally import the installed wheel.  The
    analyzer requires the other side of this comparison to be the checkout;
    do not weaken that guard or relabel an installed import as a source one.
    """
    script = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from benchmarks.analyze_phase8_cumulative_closure import analyze
probe = json.load(sys.stdin)
try:
    result = analyze(root, installed_wheel_probe=probe)
except RuntimeError as exc:
    print(json.dumps({'ok': False, 'error': str(exc)}))
else:
    print(json.dumps({'ok': True, 'report': result}, allow_nan=False))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(ROOT)],
        input=json.dumps(installed_probe, allow_nan=False),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    if not payload["ok"]:
        raise RuntimeError(payload["error"])
    return payload["report"]


@pytest.fixture(scope="module")
def analyzer():
    return _load_analyzer()


@pytest.fixture(scope="module")
def installed_probe(analyzer):
    return analyzer.build_installed_wheel_probe(ROOT)


@pytest.fixture(scope="module")
def report(analyzer, installed_probe):
    return _analyze_source(installed_probe)
