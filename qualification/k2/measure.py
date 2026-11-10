"""One CPU K2 scene/trial; run twice per side before comparing either side."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--observed", action="store_true")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None:
        if args.role != "candidate":
            parser.error("baseline must use the installed v0.4.0 wheel")
        sys.path.insert(0, str(args.source))
    import torch
    from torch.profiler import ProfilerActivity, profile

    import pssolver

    if args.role == "baseline" and pssolver.__version__ != "0.4.0":
        raise RuntimeError("baseline must be v0.4.0")
    # Both interpreters use this same qualification code, but baseline package
    # imports have already resolved to its installed wheel.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks._artifact_io import atomic_write_json
    from qualification.identity.records import tensor_record
    from qualification.identity.runtime_case import complete_simulation, initial_values
    from qualification.identity.scenarios import scenarios
    from qualification.k2.dispatch_counter import DispatchCounter
    from qualification.k2.resident_storage import ResidentStorageCounter

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    scene = next(row for row in scenarios() if row["id"] == args.scene)
    args.work.mkdir(parents=True, exist_ok=True)
    simulation = complete_simulation(scene, args.work)
    spec = pssolver.compile_simulation(simulation).application_request
    dtype = getattr(torch, scene["dtype"])
    names = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    initial = {
        name: torch.tensor(initial_values(scene)[..., i], dtype=dtype)
        for i, name in enumerate(names)
    }

    if args.role == "baseline":
        from pssolver.runtime.periodic_beris_edwards import (
            PeriodicRuntimeBuildRequest,
            build_periodic_beris_edwards_runtime,
        )

        request = PeriodicRuntimeBuildRequest(spec, initial, "cpu")

        def build():
            return build_periodic_beris_edwards_runtime(request), None

    else:
        from pssolver.configuration.legacy_periodic import translate_periodic
        from pssolver.engine import Engine

        plan = translate_periodic(spec.simulation, flavour="production", load_initial=False).plan

        def build():
            engine = Engine(plan, device="cpu", dtype=dtype, batch=1)
            return engine, engine.initial_state(initial)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as construction_profile:
        resident = ResidentStorageCounter()
        runtime, state = build()
    construction_snapshot = resident.snapshot()

    def step():
        if args.role == "baseline":
            runtime.advance(1)
        else:
            runtime.step_(state)

    def record():
        fields = runtime.fields if args.role == "baseline" else state
        return dict(spatial=tensor_record(fields.spatial), spectral=tensor_record(fields.spectral))

    initial_record = record()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as first_profile:
        step()
    first_snapshot = resident.snapshot()
    step()
    rows = []
    for index in range(3, 11):
        counter = DispatchCounter()
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            with counter:
                if args.observed:
                    if args.role == "baseline":
                        runtime.synchronize_for_observation()
                    else:
                        runtime.prepare_constraints_(state)
                        runtime.observe(state)
                step()
        events = prof.events()
        rows.append(
            dict(
                step=index,
                counts=counter.counts(),
                operators=dict(counter.operators),
                state=record(),
                supplementary_profiler=dict(
                    positive_memory_events=sum(
                        event.name == "[memory]" and event.cpu_memory_usage > 0 for event in events
                    ),
                    net_memory=sum(
                        event.cpu_memory_usage for event in events if event.name == "[memory]"
                    ),
                ),
            )
        )
    atomic_write_json(
        args.output,
        dict(
            role=args.role,
            scene=scene,
            observed=args.observed,
            python=sys.executable,
            pssolver_file=pssolver.__file__,
            pssolver_version=pssolver.__version__,
            initial=initial_record,
            rows=rows,
            resident_bytes={
                "construction": construction_snapshot["resident_bytes"],
                "after_first_step": first_snapshot["resident_bytes"],
            },
            raw_live_storages={
                "initial": resident.initial_records,
                "construction": construction_snapshot["live_storages"],
                "after_first_step": first_snapshot["live_storages"],
            },
            supplementary_profiler_resident={
                "construction": sum(
                    e.cpu_memory_usage
                    for e in construction_profile.events()
                    if e.name == "[memory]"
                ),
                "after_first_step": sum(
                    e.cpu_memory_usage
                    for p in (construction_profile, first_profile)
                    for e in p.events()
                    if e.name == "[memory]"
                ),
            },
        ),
        overwrite=False,
    )


if __name__ == "__main__":
    main()
