"""Sixteen-step read-only replay of the five retained v0.2.0 format families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import pssolver


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks._artifact_io import atomic_write_json  # noqa: E402
from qualification.identity import retained_v020_fixtures as old  # noqa: E402
from qualification.identity.checkpoint_fixtures import file_manifest  # noqa: E402
from qualification.identity.records import fingerprint  # noqa: E402


def replay(fixture: Path, work: Path) -> dict:
    before = file_manifest(fixture)
    cases = {}
    for family in ("periodic", "channel"):
        name = "functional_" + family
        runtime = (
            old.build_periodic_functional if family == "periodic" else old.build_channel_functional
        )(work / name)
        restored = runtime.checkpoint_bridge.import_checkpoint(fixture / name)
        state = restored.state
        records = [old.functional_continuation_records(state)]
        for step in range(restored.completed_steps, restored.completed_steps + 16):
            state = runtime.step(state, old._control(runtime), step)
            records.append(old.functional_continuation_records(state))
        cases[name] = records
    for name, builder, capture in (
        ("production_periodic", old.build_periodic_production, old.periodic_continuation_records),
        (
            "production_channel_complete_stress",
            old.build_channel_complete_stress,
            old.channel_complete_continuation_records,
        ),
        ("production_plane_legacy", old.build_plane_legacy, old.plane_continuation_records),
    ):
        spec, runtime = builder(work / name)
        if name == "production_periodic":
            old.restore_periodic_checkpoint(
                runtime, old.load_periodic_checkpoint(fixture / name), run_spec=spec
            )
        elif name == "production_plane_legacy":
            old.restore_plane_checkpoint(
                runtime, old.load_plane_checkpoint(fixture / name), run_spec=spec
            )
        else:
            old.load_channel_complete_stress_checkpoint(fixture / name, runtime, run_spec=spec)
        records = [capture(runtime)]
        for _ in range(16):
            if name == "production_periodic":
                old.advance_periodic(runtime)
            elif name == "production_plane_legacy":
                old.advance_plane(runtime)
            else:
                old.advance_channel(runtime)
            records.append(capture(runtime))
        cases[name] = records
    assert before == file_manifest(fixture)
    return {
        "fingerprint": fingerprint(),
        "cases": cases,
        "fixture_files_unchanged": True,
        "fixtures": before,
        "legacy_channel": "S-24 disposition only; not changed by S-23",
    }


if __name__ == "__main__":
    if not Path(pssolver.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Installed wheel required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.work.exists() or args.output.exists():
        raise FileExistsError("Historical replay paths must be new")
    atomic_write_json(args.output, replay(ROOT / "tests/fixtures/v0_2_0_checkpoints_v1", args.work))
