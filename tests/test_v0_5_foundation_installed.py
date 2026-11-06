"""The installed probe is explicit; absence of its interpreter is a failed prerequisite."""

import json
import os
import subprocess
from pathlib import Path

import pssolver


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_record_and_installed_metadata_are_identical(tmp_path):
    active = json.loads((ROOT / "notes/pssolver_active_version_record.json").read_text())
    version = json.loads((ROOT / active["record"]).read_text())[active["version_field"]]
    result = subprocess.run(
        [
            os.environ["PSSOLVER_INSTALLED_PYTHON"],
            "-I",
            "-B",
            "-c",
            "import importlib.metadata as m,json,pssolver,sys; "
            "print(json.dumps(dict(runtime=pssolver.__version__,metadata=m.version('pssolver'),"
            "origin=pssolver.__file__,prefix=sys.prefix)))",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["runtime"] == report["metadata"] == pssolver.__version__ == version
    assert Path(report["origin"]).is_relative_to(Path(report["prefix"]))
    assert "site-packages" in Path(report["origin"]).parts
