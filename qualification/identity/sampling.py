"""Explicit qualification events; observation and checkpoint are separate phases."""

from __future__ import annotations

import gc
from pathlib import Path


def sample(case, root: Path, step: int, *, restart_from: int | None = None) -> dict:
    """Replay the same event schedule, without rewriting runtime flags or tensors."""
    row = {"before_observation": case.record()}
    if case.scene["observe"]:
        row["observations"] = case.observe()
    row["after_observation"] = case.record()
    if step in case.scene["restart_steps"]:
        name = f"restart_{step}" if restart_from is None else f"replay_{restart_from}_{step}"
        case.export(root / name)
    row["after_checkpoint"] = case.record()
    if step < case.scene["steps"]:
        row["before_explicit_rhs"] = case.step()
        row["after_step"] = case.record()
    return row


def restart_contract(continuous: list[dict], resumed: list[dict], start: int) -> dict:
    """Check exact field bytes and clocks; expose, never normalize, solver metadata."""
    differences = []
    metadata = []
    for offset, actual in enumerate(resumed):
        expected = continuous[start + offset]
        phases = ["after_checkpoint"]
        if "after_step" in expected:
            phases += ["before_explicit_rhs", "after_step"]
        for phase in phases:
            left, right = expected[phase], actual[phase]
            if left is None and right is None:
                continue
            if left["step"] != right["step"] or left["tensors"] != right["tensors"]:
                differences.append({"step": start + offset, "phase": phase, "kind": "tensors"})
            for key in ("step_count", "refresh_count", "refresh_interval"):
                if left.get("progress", {}).get(key) != right.get("progress", {}).get(key):
                    differences.append({"step": start + offset, "phase": phase, "kind": key})
            for key in ("progress", "solver"):
                if left.get(key) != right.get(key):
                    metadata.append(
                        {
                            "step": start + offset,
                            "phase": phase,
                            "kind": key,
                            "continuous": left.get(key),
                            "resumed": right.get(key),
                        }
                    )
    return {
        "field_bytes_and_clocks_equal": not differences,
        "differences": differences,
        "raw_metadata_differences": metadata,
    }


def trace(scene: dict, root: Path, *, case_factory=None) -> dict:
    """Record one continuous run and resumptions with identical event schedules."""
    from qualification.identity.runtime_case import RuntimeCase

    factory = RuntimeCase if case_factory is None else case_factory
    case = factory(scene, root)
    identity = case.identity()
    rows = [sample(case, root, step) for step in range(scene["steps"] + 1)]
    del case
    gc.collect()
    resumptions, contracts = {}, {}
    for start in scene["restart_steps"]:
        restored = factory(scene, root)
        restored.restore(root / f"restart_{start}", start)
        continued = [
            sample(restored, root, step, restart_from=start)
            for step in range(start, scene["steps"] + 1)
        ]
        resumptions[str(start)] = continued
        contracts[str(start)] = restart_contract(rows, continued, start)
        del restored
        gc.collect()
    return {
        "identity": identity,
        "steps": rows,
        "restarts": resumptions,
        "restart_contracts": contracts,
    }
