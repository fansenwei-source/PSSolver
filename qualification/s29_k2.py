"""Capture step-local K2 traces without altering the accepted dispatch counter."""

import argparse
import json
import os
import sys
from pathlib import Path


def measure(root, work):
    sys.path.insert(0, str(root))
    import torch

    import pssolver as public
    from qualification.identity.scenarios import scenarios
    from qualification.identity.scenes_v05 import initial_values, simulation_v05
    from qualification.k2.dispatch_counter import DispatchCounter

    assert Path(public.__file__).resolve().is_relative_to(root)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = {}
    for family in ("periodic", "plane", "channel"):
        scene = next(
            row
            for row in scenarios()
            if row["family"] == family and row["dtype"] == "float64" and not row["options"]
        )
        compiled = public.compile_simulation(simulation_v05(scene, work / family))
        array = initial_values(scene)
        for mode in ("step", "prepare_observe_step"):
            engine = compiled._run.assembled.engine(device="cpu", dtype=torch.float64)
            names = engine.components[: engine.evolved_count]
            state = engine.initial_state(
                {name: torch.tensor(array[..., i]) for i, name in enumerate(names)}
            )
            rows = []
            for i in range(1, 11):
                with DispatchCounter() as counter:
                    if mode == "prepare_observe_step":
                        engine.prepare_constraints_(state)
                        engine.observe(state)
                    engine.step_(state)
                rows.append({"step": i, **counter.counts(), "operators": dict(counter.operators)})
            result[family + "/" + mode] = rows
    assert not torch.cuda.is_initialized()
    return result


def compare(paths):
    a, a2, b, b2 = (json.loads(path.read_bytes()) for path in paths)
    assert a.keys() == a2.keys() == b.keys() == b2.keys()
    if a != a2 or b != b2:
        return {"deterministic": False, "passed": False}
    exceeded = []
    for name, old in a.items():
        for x, y in zip(old, b[name], strict=True):
            assert x["step"] == y["step"]
            for key in ("transforms", "allocations"):
                if y[key] > x[key]:
                    exceeded.append((name, x["step"], key, x[key], y[key]))
    return {"deterministic": True, "exceeded": exceeded, "passed": not exceeded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--compare", type=Path, nargs=4)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = compare(args.compare) if args.compare else measure(args.root, args.work)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return int(not value.get("passed", True))


if __name__ == "__main__":
    raise SystemExit(main())
