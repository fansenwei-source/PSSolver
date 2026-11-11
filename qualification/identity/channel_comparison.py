"""Channel probes for the Engine; leave both v0.4 reference drivers untouched."""

import copy
from pathlib import Path

import torch

from pssolver import compile_simulation
from pssolver.configuration.legacy_channel import load_channel_initial_q, translate_channel
from pssolver.engine.engine import Engine
from pssolver.io import engine_state
from pssolver.runtime.legacy_entry import restore_channel_state
from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import complete_simulation


class ChannelEngineCase:
    """Observe the same events as the S-28a design dry run."""

    def __init__(self, scene, root):
        self.scene = scene
        self.index = 0
        simulation = complete_simulation(scene, root)
        self.configuration = simulation.identity_metadata()
        self.spec = compile_simulation(simulation).application_request
        values, _, _ = load_channel_initial_q(self.spec)
        self.engine = Engine(
            translate_channel(self.spec), device="cpu", dtype=getattr(torch, scene["dtype"])
        )
        self.diagnostics = {}
        self.state = self.engine.initial_state(values, diagnostics=self.diagnostics)

    def identity(self):
        path = self.spec.runtime_path
        return {
            "configuration": self.configuration,
            "runtime": {
                "adapter": "Engine",
                "requested": path,
                "effective": path,
                "fallback_used": False,
                "pressure_gauge": "zero_mean",
                "velocity_nullspace": "absent_due_to_two_bounded_no_slip_axes",
                "pressure_warm_start": True,
                "pressure_solver_policy": (
                    self.engine.constraint.term.pressure_solver.pressure_convergence_policy()
                ),
                "pressure_diagnostics_schema_version": 2,
            },
        }

    def prepare(self):
        self.engine.prepare_constraints_(self.state, diagnostics=self.diagnostics)

    def record(self):
        fields = self.engine.field_view(self.state)
        tensors = {
            name + suffix: tensor_record(fields[name + suffix])
            for name in self.engine.components
            for suffix in ("", ".hat")
        }
        guess = self.state.solver.get("pressure_guess")
        if guess is not None:
            tensors["pressure_guess"] = tensor_record(guess)
        return {
            "step": self.index,
            "tensors": tensors,
            "progress": {
                "step_count": self.state.refresh_phase[0],
                "refresh_count": self.state.refresh_phase[1],
                "refresh_interval": self.engine.plan.integrator.refresh_interval,
                "constraints_current": self.state.constrained_current,
            },
        }

    def observe(self):
        self.prepare()
        fields = self.engine.field_view(self.state)
        return {name: tensor_record(fields[name]) for name in self.engine.components}

    def step(self):
        self.prepare()
        self.engine.step_(self.state, diagnostics=self.diagnostics)
        self.index += 1

    def export(self, path):
        self.prepare()
        engine_state.write(
            path,
            self.state,
            plan_identity_sha256=self.engine.plan.identity_sha256,
            plan_metadata=self.engine.plan.to_metadata(),
            refresh_interval=self.engine.plan.integrator.refresh_interval,
            run_provenance={"channel_constraint_diagnostics": self.diagnostics},
        )

    def restore(self, path, step, *, legacy=False):
        if legacy:
            restored = restore_channel_state(self.engine, self.state, path, run_spec=self.spec)
            assert restored == step
        else:
            expected = {
                "plan_identity_sha256": self.engine.plan.identity_sha256,
                "dtype": self.engine.dtype,
                "batch": self.engine.batch,
            }
            metadata = engine_state.read_header(path, **expected)
            self.diagnostics = metadata["run_provenance"]["channel_constraint_diagnostics"]
            self.state = engine_state.read(path, **expected, device=self.engine.device)
            self.engine._restore_channel_state_layout(self.state)
        self.index = step


def sample(case, scene):
    """Capture only the phases declared in the authority's dry-run records."""
    row = {"before_observation": case.record()}
    if scene["observe"]:
        row["observations"] = case.observe()
    row["after_observation"] = case.record()
    return row


def trace_channel(scene, root, *, case_factory=ChannelEngineCase):
    """Preserve raw field bytes and both restart endpoints."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    case = case_factory(scene, root)
    steps = []
    for index in range(scene["steps"] + 1):
        steps.append(sample(case, scene))
        if index in scene["restart_steps"]:
            case.export(root / f"restart_{index}")
            if not scene["observe"]:
                # The sealed dry run sampled this phase after checkpoint I/O
                # when no observation took place.
                steps[-1]["after_observation"] = case.record()
        if index < scene["steps"]:
            case.step()
    resumed = {}
    for start in scene["restart_steps"]:
        continuation = case_factory(scene, root)
        continuation.restore(root / f"restart_{start}", start)
        rows = []
        for index in range(start, scene["steps"] + 1):
            rows.append(sample(continuation, scene)["after_observation"])
            if index < scene["steps"]:
                continuation.step()
        resumed[str(start)] = rows
    return {"identity": case.identity(), "steps": steps, "restarts": resumed}


def comparable_channel(trace):
    """Apply only the two exact S-28a raw-trace exclusions."""
    result = copy.deepcopy(trace)
    del result["identity"]["runtime"]["adapter"]
    for step in result["steps"]:
        for phase in ("before_observation", "after_observation"):
            del step[phase]["progress"]["constraints_current"]
    for rows in result["restarts"].values():
        for row in rows:
            del row["progress"]["constraints_current"]
    return result


def channel_tensor_progress(row):
    """Compare the field/clock contract independently of solver cache representation."""
    return {
        "step": row["step"],
        "tensors": row["tensors"],
        "progress": {
            key: row["progress"][key]
            for key in (
                "step_count",
                "refresh_count",
                "refresh_interval",
            )
        },
    }
