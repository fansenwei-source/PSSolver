"""S-28b Plan and trajectory gates against sealed S-23 fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

import pssolver as pub
from qualification.identity.records import tensor_record
from qualification.identity.sampling import trace
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import functional_plan, initial_values, simulation_v05


_CACHE_SCENES = frozenset(
    {
        "channel_default_float64_relative_tolerance",
        "channel_default_float64_relative_tolerance_float32_floor",
        "channel_default_float32_relative_tolerance_float32_floor",
    }
)
_CACHE_KEYS = (
    "operator_action",
    "operator_action_matches_guess",
    "operator_action_working_exponent",
)


class Case05:
    """RuntimeCaseV2 interface (S-23 event protocol of qualification/identity/sampling.py) on the 0.5 public path:
    compile_simulation -> prepare_run; restarts through prepare_run(Output(restart_from=engine_state dir))."""

    def __init__(self, scene, root, *, expected_plan):
        self.expected_plan = expected_plan
        self.scene, self.family, self.root, self.index = scene, scene["family"], root, 0
        self._prepare(output_dir=root / "output", steps=scene["steps"], restart_from=None)

    def _prepare(self, *, output_dir, steps, restart_from):
        from pssolver.runtime.simulation_run import build_run_metadata, prepare_run

        compiled = pub.compile_simulation(
            simulation_v05(
                self.scene,
                self.root,
                output_dir=output_dir,
                steps=steps,
                restart_from=restart_from,
            )
        )
        assert compiled.plan_identity_sha256 == self.expected_plan
        injected = None
        if self.family == "plane":
            values = initial_values(self.scene)
            injected = {
                name: torch.tensor(values[..., i])
                for i, name in enumerate(("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"))
            }
        prepared = prepare_run(
            compiled,
            metadata=build_run_metadata(compiled),
            device=torch.device("cpu"),
            dtype=getattr(torch, self.scene["dtype"]),
            initial_values=injected,
        )
        self.engine, self.state = prepared.engine, prepared.state
        self.last_diagnostics = dict(prepared.construction_diagnostics)
        self.start_step = prepared.start_step

    def restore(self, path, step):
        self._prepare(
            output_dir=self.root / f"output_restart_{step}",
            steps=self.scene["steps"] - step,
            restart_from=path,
        )
        assert self.start_step == step, (self.start_step, step)
        self.index = step

    def identity(self):
        return {}  # /identity is compared by W1, W5 and W6 (design §8)

    def prepare(self):
        if self.family == "channel":
            self.engine.prepare_constraints_(self.state, diagnostics=self.last_diagnostics)
        else:
            self.engine.prepare_constraints_(
                self.state,
                controls={"alpha": self.engine.default_controls["alpha"]},
                diagnostics=self.last_diagnostics,
            )

    def record(self):
        fields = self.engine.field_view(self.state)
        tensors = {
            n + s: tensor_record(fields[n + s])
            for n in self.engine.components
            for s in ("", ".hat")
        }
        guess = (
            self.state.solver.get("pressure_guess") if isinstance(self.state.solver, dict) else None
        )
        if guess is not None:
            tensors["pressure_guess"] = tensor_record(guess)
        key = "last_pressure_iterations" if self.family == "plane" else "pressure_iterations"
        solver = {
            "operator_action": None,
            "operator_action_working_exponent": None,
            "operator_action_matches_guess": False,
            "pressure_iterations": self.last_diagnostics.get(key),
        }
        if (
            self.family == "channel"
        ):  # S-28 r1.6 V1 map: warm_operator_action <-> pressure_guess_action
            action = self.state.solver.get("pressure_guess_action")
            solver = {
                "operator_action": None if action is None else tensor_record(action),
                "operator_action_working_exponent": self.state.solver.get(
                    "pressure_guess_action_exponent"
                ),
                "operator_action_matches_guess": action is not None,
                "pressure_iterations": self.last_diagnostics.get("achieved_iteration_count"),
            }
        return {
            "step": self.index,
            "tensors": tensors,
            "progress": {
                "step_count": self.state.refresh_phase[0],
                "refresh_count": self.state.refresh_phase[1],
                "refresh_interval": self.engine.plan.integrator.refresh_interval,
                "constraints_current": self.state.constrained_current,
            },
            "solver": solver,
        }

    def observe(self):
        self.prepare()
        fields = self.engine.field_view(self.state)
        return {n: tensor_record(fields[n]) for n in self.engine.components}

    def step(self):
        self.prepare()
        before = self.record()
        self.engine.step_(self.state, diagnostics=self.last_diagnostics)
        self.index += 1
        return before

    def export(self, path):
        from pssolver.io import engine_state

        self.prepare()
        engine_state.write(
            path,
            self.state,
            plan_identity_sha256=self.engine.plan.identity_sha256,
            plan_metadata=self.engine.plan.to_metadata(),
            refresh_interval=self.engine.plan.integrator.refresh_interval,
            run_provenance={"channel_constraint_diagnostics": self.last_diagnostics}
            if self.family == "channel"
            else {},
        )


def _portable_trace(trace_record):
    """Drop only the pre-existing transient constraint-current flag."""
    value = deepcopy({k: trace_record[k] for k in ("steps", "restarts", "restart_contracts")})
    for rows in (value["steps"], *value["restarts"].values()):
        for row in rows:
            for item in row.values():
                if isinstance(item, dict) and "progress" in item:
                    item["progress"].pop("constraints_current", None)
    for contract in value["restart_contracts"].values():
        kept = []
        for item in contract["raw_metadata_differences"]:
            if item["kind"] == "progress":
                for key in ("continuous", "resumed"):
                    item[key].pop("constraints_current", None)
                if item["continuous"] == item["resumed"]:
                    continue
            kept.append(item)
        contract["raw_metadata_differences"] = kept
    return value


def _use_continuous_cache(observed, expected):
    """Require resumed cache leaves to match the candidate's own continuous run."""
    for start, rows in observed["restarts"].items():
        expected_rows = expected["restarts"][start]
        for offset, row in enumerate(rows):
            step = int(start) + offset
            continuous = observed["steps"][step]
            for phase, item in row.items():
                if not isinstance(item, dict) or "solver" not in item:
                    continue
                # At the first resumed event the stored state is already
                # past the continuous row's observation/checkpoint boundary.
                # These early samples must see the cache saved at that boundary.
                continuous_phase = (
                    "after_checkpoint"
                    if offset == 0 and phase in ("before_observation", "after_observation")
                    else phase
                )
                wanted = continuous[continuous_phase]["solver"]
                got = item["solver"]
                for key in _CACHE_KEYS:
                    if type(wanted[key]) is not type(got[key]) or wanted[key] != got[key]:
                        raise ValueError(f"D-0069 cache mismatch: {start}/{step}/{phase}/{key}")
                    expected_rows[offset][phase]["solver"][key] = wanted[key]
        # Raw metadata rows are difference records, not an additional numerical
        # comparison. Substitute only the three authorized resumed cache leaves.
        contract = expected["restart_contracts"][start]
        remaining = []
        for item in contract["raw_metadata_differences"]:
            if item["kind"] == "solver":
                solver = observed["steps"][item["step"]][item["phase"]]["solver"]
                for key in _CACHE_KEYS:
                    item["resumed"][key] = solver[key]
                if item["continuous"] == item["resumed"]:
                    continue
            remaining.append(item)
        contract["raw_metadata_differences"] = remaining


def _leaves(value, path=()):
    if isinstance(value, dict) and value:
        for name, item in value.items():
            yield from _leaves(item, (*path, name))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def compare_trajectory(sid, candidate, sealed):
    """Compare all portable leaves; only D-0069 changes a comparison source."""
    observed = _portable_trace(candidate)
    expected = _portable_trace(sealed)
    if sid in _CACHE_SCENES:
        _use_continuous_cache(observed, expected)
    left, right = dict(_leaves(expected)), dict(_leaves(observed))
    missing = object()
    differences = [
        "/" + "/".join(map(str, path))
        for path in sorted(left.keys() | right.keys(), key=str)
        if type(left.get(path, missing)) is not type(right.get(path, missing))
        or left.get(path, missing) != right.get(path, missing)
    ]
    return {"leaves": len(left), "differences": differences}


def main():
    """Run selected accepted scenarios from the r2 authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "trajectory"))
    parser.add_argument("reference", type=Path, help="sealed matrix_baseline_first directory")
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("scenes", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    table = Path(__file__).resolve().parent / "s28b_tables.json"
    plans = json.loads(table.read_bytes())["plan_identity_reference"]
    selected = set(args.scenes)
    rows = {}
    for scene in scenarios():
        sid = scene["id"]
        if sid not in plans or (selected and sid not in selected):
            continue
        functional = scene["family"].startswith("functional_")
        if args.mode == "trajectory" and functional:
            continue
        root = args.work / sid
        root.mkdir(parents=True)
        expected = plans[sid]["plan_identity_sha256"]
        try:
            if args.mode == "plan":
                plan = (
                    functional_plan(scene, root)
                    if functional
                    else pub.compile_simulation(simulation_v05(scene, root)).plan_identity_sha256
                )
                rows[sid] = {"plan": plan, "expected": expected, "passed": plan == expected}
            else:
                sealed = json.loads((args.reference / f"{sid}.json").read_bytes())
                case = partial(Case05, expected_plan=expected)
                result = compare_trajectory(
                    sid,
                    trace(scene, root, case_factory=case),
                    sealed["trace"],
                )
                rows[sid] = {**result, "passed": not result["differences"]}
        except (OSError, TypeError, ValueError, RuntimeError, AssertionError, KeyError) as exc:
            rows[sid] = {"passed": False, "type": type(exc).__name__, "error": str(exc)}
    expected_count = len(selected) if selected else (104 if args.mode == "plan" else 98)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    failed = [sid for sid, row in rows.items() if not row["passed"]]
    print(json.dumps({"expected": expected_count, "ran": len(rows), "failed": failed}))
    return int(len(rows) != expected_count or bool(failed))


if __name__ == "__main__":
    sys.exit(main())
