"""Record the S-26 complete aten trace, including exact output bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None:
        sys.path.insert(0, str(args.source))
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    import pssolver
    from pssolver.configuration.legacy_periodic import translate_periodic
    from pssolver.engine import Engine

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qualification.identity.runtime_case import complete_simulation, initial_values
    from qualification.identity.scenarios import scenarios

    def metadata(value, *, output=False):
        if isinstance(value, torch.Tensor):
            row = dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                storage_offset=value.storage_offset(),
                contiguous=value.is_contiguous(),
            )
            if output:
                array = value.detach().resolve_conj().resolve_neg().contiguous().numpy()
                row["sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
            return row
        if isinstance(value, (list, tuple)):
            return [metadata(item, output=output) for item in value]
        if isinstance(value, dict):
            return {str(key): metadata(item, output=output) for key, item in value.items()}
        return repr(value)

    class Trace(TorchDispatchMode):
        def __init__(self):
            self.rows = []
            self.stacks = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            inputs = metadata((args, kwargs))
            result = func(*args, **kwargs)
            self.rows.append(
                dict(aten=str(func), inputs=inputs, output=metadata(result, output=True))
            )
            self.stacks.append(traceback.format_stack())
            return result

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    args.output.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    scene = next(row for row in scenarios() if row["id"] == args.scene)
    provenance = dict(
        provider=pssolver.__file__,
        version=pssolver.__version__,
        scene=scene,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        declaration = complete_simulation(scene, args.work).specification
        translation = translate_periodic(
            declaration,
            flavour=(
                "functional_1_1" if scene["family"] == "functional_periodic" else "production"
            ),
            load_initial=False,
        )
    except (ValueError, TypeError) as error:
        (args.output / "plan.json").write_text(
            json.dumps(dict(rejected=type(error).__name__, message=str(error)), sort_keys=True)
            + "\n"
        )
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return
    plan = translation.plan
    (args.output / "plan.json").write_text(json.dumps(plan.to_metadata(), sort_keys=True) + "\n")
    raw = initial_values(scene)
    dtype = getattr(torch, scene["dtype"])
    initial = {
        name: torch.tensor(raw[..., index], dtype=dtype)
        for index, name in enumerate(("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"))
    }
    controls = (
        {"alpha": torch.zeros(scene["shape"], dtype=dtype)}
        if scene["family"] == "functional_periodic"
        else None
    )
    traces = {}
    with Trace() as trace:
        engine = Engine(plan, device="cpu", dtype=dtype)
    traces["construct"] = trace
    with Trace() as trace:
        state = engine.initial_state(initial)
    traces["initial_state"] = trace
    with Trace() as trace:
        state = engine.prepare_constraints(state, controls)
    traces["prepare_constraints"] = trace
    for stage in ("pure", "inplace"):
        with Trace() as trace:
            for _ in range(2):
                if stage == "pure":
                    state = engine.step(state, controls)
                else:
                    engine.step_(state, controls)
        traces[stage] = trace
    for name, trace in traces.items():
        (args.output / (name + ".json")).write_text(json.dumps(trace.rows, sort_keys=True) + "\n")
        (args.output / (name + ".stacks.json")).write_text(json.dumps(trace.stacks) + "\n")
    provenance["counts"] = {name: len(trace.rows) for name, trace in traces.items()}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
