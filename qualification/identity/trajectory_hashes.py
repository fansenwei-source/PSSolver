"""Run S-23 scene JSON against an installed wheel, never a source-shadow package.

Launch with the selected venv's ``python -I -B``. Results are CPU reference
measurements, not GPU qualification. All runs must share the same work path.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

import pssolver


# Import the installed runtime first; only qualification and shared artifact I/O
# are then sourced from this checkout. Every loaded runtime module is checked.
PACKAGE_ROOT = Path(pssolver.__file__).resolve().parent
if not PACKAGE_ROOT.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError(f"Expected venv-local installed package, got {PACKAGE_ROOT}")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks._artifact_io import atomic_write_json  # noqa: E402
from qualification.identity.records import fingerprint  # noqa: E402
from qualification.identity.runtime_case import RuntimeCase  # noqa: E402
from qualification.identity.scenarios import scenarios  # noqa: E402


def assert_origins() -> dict[str, str]:
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "pssolver" or name.startswith("pssolver."))
        and getattr(module, "__file__", None)
    }
    if any(not Path(path).is_relative_to(PACKAGE_ROOT) for path in origins.values()):
        raise RuntimeError("Source shadow import detected")
    return origins


def trace(scene: dict, root: Path) -> dict:
    """Record one continuous run and independent checkpoint resumptions."""
    case = RuntimeCase(scene, root)
    identity = case.identity()
    checkpoints = {}
    rows = []
    for step in range(scene["steps"] + 1):
        row = {"before_observation": case.record()}
        if scene["observe"]:
            row["observations"] = case.observe()
        if step in scene["restart_steps"]:
            path = root / f"restart_{step}"
            case.export(path)
            checkpoints[step] = path
        row["after_observation"] = case.record()
        rows.append(row)
        if step < scene["steps"]:
            case.step()
    del case
    gc.collect()
    resumptions = {}
    for start, path in checkpoints.items():
        restored = RuntimeCase(scene, root)
        restored.restore(path, start)
        continued = []
        for step in range(start, scene["steps"] + 1):
            if scene["observe"]:
                restored.observe()
            continued.append(restored.record())
            if step < scene["steps"]:
                restored.step()
        resumptions[str(start)] = continued
        del restored
        gc.collect()
    return {"identity": identity, "steps": rows, "restarts": resumptions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.work.exists():
        raise FileExistsError("Work and output must initially be absent")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only execution must not initialize CUDA")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    matrix = json.loads(args.scenes.read_text()) if args.scenes else scenarios()
    if args.scene_id:
        matrix = [row for row in matrix if row["id"] == args.scene_id]
        if not matrix:
            raise ValueError("Unknown scene id")
    args.output.mkdir(parents=True)
    # This exact newly-created subtree is the only disposable computation root.
    args.work.mkdir(parents=True)
    atomic_write_json(
        args.output / "environment.json",
        {
            "fingerprint": fingerprint(),
            "package_root": str(PACKAGE_ROOT),
            "version": pssolver.__version__,
            "sys_path": sys.path,
            "scene_count": len(matrix),
            "discovery_only": args.discover,
        },
    )
    for scene in matrix:
        work = args.work / scene["id"]
        work.mkdir()
        result = {"scene": scene}
        try:
            try:
                case = RuntimeCase(scene, work)
            except (ValueError, TypeError, NotImplementedError) as error:
                result.update(
                    disposition="rejected",
                    exception_type=f"{type(error).__module__}.{type(error).__name__}",
                    message=str(error),
                )
            else:
                result.update(disposition="accepted", identity=case.identity())
                del case
                gc.collect()
                if not args.discover:
                    result["trace"] = trace(scene, work)
            assert_origins()
            atomic_write_json(args.output / (scene["id"] + ".json"), result)
            print(scene["id"], result["disposition"], flush=True)
        except Exception as error:
            atomic_write_json(
                args.output / "STOPPED.json",
                {"scene": scene, "exception": type(error).__name__, "message": str(error)},
            )
            raise
        # Numeric output records are already retained. Failed work is not deleted.
        shutil.rmtree(work)
    atomic_write_json(args.output / "origins.json", assert_origins())
    args.work.rmdir()


if __name__ == "__main__":
    main()
