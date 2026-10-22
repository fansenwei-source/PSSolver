"""Verify that the released v0.2.2 wheel rejects v0.3 S-05 carriers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError("qualification JSON readback differs")


def _child_program() -> str:
    return r'''
import json
from pathlib import Path
import sys

wheel, root = sys.argv[1:]
sys.path.insert(0, wheel)

import pssolver
from pssolver.experimental import load_shadow_checkpoint
from pssolver.workflows import (
    load_channel_checkpoint,
    load_finite_q_anchoring_checkpoint,
    load_plane_checkpoint,
)

loaders = {
    "plane": load_plane_checkpoint,
    "channel": load_channel_checkpoint,
    "finite_q": load_finite_q_anchoring_checkpoint,
    "shadow": load_shadow_checkpoint,
}
results = {}
for kind, loader in loaders.items():
    try:
        loader(Path(root) / kind)
    except Exception as exc:
        results[kind] = {
            "accepted": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        results[kind] = {"accepted": True}

print(json.dumps({
    "pssolver_version": pssolver.__version__,
    "pssolver_import": pssolver.__file__,
    "sys_path": sys.path,
    "results": results,
}, allow_nan=False, sort_keys=True))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wheel = args.wheel.expanduser().resolve()
    repository = args.repository.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"qualification output already exists: {output}")
    if _sha256(wheel) != args.expected_wheel_sha256:
        raise ValueError("v0.2.2 wheel SHA-256 differs")
    output.mkdir(parents=True)
    checkpoint_root = output / "v0_3_checkpoints"
    checkpoint_root.mkdir()

    support_path = (
        repository / "tests/test_v0_3_s05_checkpoint_metadata_integrity.py"
    )
    support = runpy.run_path(str(support_path))
    write_current = support["_write_current"]
    for kind in ("plane", "channel", "finite_q", "shadow"):
        write_current(kind, checkpoint_root / kind, output)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _child_program(),
            str(wheel),
            str(checkpoint_root),
        ],
        cwd=output,
        check=False,
        capture_output=True,
        text=True,
    )
    (output / "v0_2_2_reader.stdout").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output / "v0_2_2_reader.stderr").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError("v0.2.2 reader subprocess failed before adjudication")
    child = json.loads(completed.stdout)
    expected_messages = {
        "plane": "unsupported Plane workflow checkpoint version",
        "channel": "unsupported Channel checkpoint version",
        "finite_q": "finite-Q checkpoint metadata schema differs",
        "shadow": "unsupported shadow checkpoint format version",
    }
    for kind, expected in expected_messages.items():
        result = child["results"][kind]
        if result["accepted"] or expected not in result["message"]:
            raise RuntimeError(f"v0.2.2 did not explicitly reject {kind}")
    if child["pssolver_version"] != "0.2.2":
        raise RuntimeError("reader did not import PSSolver v0.2.2")
    if not child["pssolver_import"].startswith(str(wheel)):
        raise RuntimeError("reader PSSolver import did not come from the wheel")

    report = {
        "schema": "pssolver.v0_3.s05.v022_forward_rejection.v1",
        "classification": "PASS_V0_3_S05_V022_FORWARD_REJECTION",
        "qualification_complete": True,
        "wheel": {
            "path": str(wheel),
            "sha256": _sha256(wheel),
            "version": child["pssolver_version"],
            "import_path": child["pssolver_import"],
            "sys_path": child["sys_path"],
        },
        "new_formats": {
            "plane": 3,
            "channel": 3,
            "finite_q": 2,
            "shadow": 2,
        },
        "results": child["results"],
        "accepted_count": sum(
            int(value["accepted"]) for value in child["results"].values()
        ),
        "explicit_rejection_count": 4,
        "source_shadow_import": False,
    }
    _write_json_new(output / "qualification.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
