"""Authentic installed-v0.4 writers and read-only 16-step continuation records.

Existing v0.2/rc fixtures remain historical artifacts, never re-labelled outputs.
The six new fixtures cover current production and functional format families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import pssolver


PACKAGE = Path(pssolver.__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._artifact_io import atomic_write_json  # noqa: E402
from qualification.identity.records import fingerprint  # noqa: E402
from qualification.identity.runtime_case import RuntimeCase  # noqa: E402
from qualification.identity.scenarios import scenarios  # noqa: E402


def fixture_scenes() -> list[dict]:
    return [
        {**row, "shape": [6, 6, 4], "steps": 18, "restart_steps": []}
        for row in scenarios()
        if row["dtype"] == "float64" and not row["options"] and row["observe"]
    ]


def file_manifest(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def continue_case(
    scene: dict, checkpoint: Path, work: Path, *, case_factory=RuntimeCase
) -> list[dict]:
    work.mkdir(parents=True, exist_ok=False)
    case = case_factory(scene, work)
    case.restore(checkpoint, 2)
    rows = [case.record()]
    for _ in range(16):
        case.step()
        rows.append(case.record())
    return rows


def main():
    if not PACKAGE.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Checkpoint fixtures require a venv-installed wheel")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.work.exists() or args.output.exists():
        raise FileExistsError("Continuation work/output must be new")
    if args.generate:
        if pssolver.__version__ != "0.4.0":
            raise RuntimeError("Only released 0.4.0 may create the baseline fixtures")
        args.fixtures.mkdir(parents=True, exist_ok=False)
        args.work.mkdir(parents=True)
        for scene in fixture_scenes():
            work = args.work / (scene["id"] + "_generation")
            work.mkdir()
            case = RuntimeCase(scene, work)
            for _ in range(2):
                case.step()
            case.export(args.fixtures / scene["id"])
        atomic_write_json(
            args.fixtures / "manifest.json",
            {
                "schema": 1,
                "writer_version": pssolver.__version__,
                "release_commit": "4c12228aba01e8483e80d28a1471ab72384d5fda",
                "wheel_sha256": "c5cf3e26af8bdde27287f6129fe5939d6f202fb627dcdc9a66c19d62871dd873",
                "scenes": fixture_scenes(),
                "files": file_manifest(args.fixtures),
                "fingerprint": fingerprint(),
                "continuation_steps": 16,
            },
        )
    before = file_manifest(args.fixtures)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    result = {"fingerprint": fingerprint(), "cases": {}}
    for scene in manifest["scenes"]:
        result["cases"][scene["id"]] = continue_case(
            scene, args.fixtures / scene["id"], args.work / scene["id"]
        )
    if before != file_manifest(args.fixtures):
        raise RuntimeError("Read-only checkpoint changed")
    result["fixture_files_unchanged"] = True
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
