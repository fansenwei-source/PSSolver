"""Explicit S-25 production Periodic adapter; the frozen v1 driver stays intact."""

import json
from pathlib import Path

import torch

from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import RuntimeCase, complete_simulation


class RuntimeCaseV2(RuntimeCase):
    """Use Engine/State only for production Periodic, and v1 for other families."""

    def __init__(self, scene: dict, root: Path):
        self.engine_periodic = scene["family"] == "periodic"
        if not self.engine_periodic:
            super().__init__(scene, root)
            return
        from pssolver.configuration.legacy_periodic import translate_periodic
        from pssolver.engine import Engine

        self.scene, self.family, self.functional = scene, "periodic", False
        self.index = 0
        simulation = complete_simulation(scene, root)
        self.configuration_identity = simulation.identity_metadata()
        self.translation = translate_periodic(
            simulation.specification, flavour="production", load_initial=True
        )
        self.spec = self.translation.run_spec
        self.engine = Engine(
            self.translation.plan, device="cpu", dtype=getattr(torch, scene["dtype"])
        )
        self.last_diagnostics = {}
        self.state = self.engine.initial_state(
            self.translation.initial_values, diagnostics=self.last_diagnostics
        )

    def prepare(self):
        self.engine.prepare_constraints_(
            self.state, controls={"alpha": self.engine.default_controls["alpha"]},
            diagnostics=self.last_diagnostics,
        )

    def step(self):
        if not self.engine_periodic:
            return super().step()
        self.prepare()
        before = self.record()
        self.engine.step_(self.state, diagnostics=self.last_diagnostics)
        self.index += 1
        return before

    def observe(self):
        if not self.engine_periodic:
            return super().observe()
        self.prepare()
        fields = self.engine.field_view(self.state)
        return {name: tensor_record(fields[name]) for name in self.engine.components}

    def record(self):
        if not self.engine_periodic:
            return super().record()
        fields = self.engine.field_view(self.state)
        return {
            "step": self.index,
            "tensors": {
                name + suffix: tensor_record(fields[name + suffix])
                for name in self.engine.components
                for suffix in ("", ".hat")
            },
            "progress": {
                "step_count": self.state.refresh_phase[0],
                "refresh_count": self.state.refresh_phase[1],
                "refresh_interval": self.engine.plan.integrator.refresh_interval,
                "constraints_current": self.state.constrained_current,
            },
            "solver": {
                "operator_action": None,
                "operator_action_working_exponent": None,
                "operator_action_matches_guess": False,
                "pressure_iterations": self.last_diagnostics["pressure_iterations"],
            },
        }

    def export(self, path):
        if not self.engine_periodic:
            return super().export(path)
        from pssolver.io import engine_state

        self.prepare()
        engine_state.write(
            path,
            self.state,
            plan_identity_sha256=self.engine.plan.identity_sha256,
            plan_metadata=self.engine.plan.to_metadata(),
            refresh_interval=self.engine.plan.integrator.refresh_interval,
            run_provenance={},
        )

    def restore(self, path, step):
        if not self.engine_periodic:
            return super().restore(path, step)
        from pssolver.io import engine_state
        from pssolver.io.legacy.periodic_workflow import load_periodic_checkpoint
        from pssolver.runtime.legacy_entry import restore_periodic_state

        # The filename is the explicit format discriminator, not a source fallback.
        metadata = json.loads((path / "checkpoint.json").read_text())
        if metadata.get("format_kind") == "engine_state":
            self.state = engine_state.read(
                path,
                plan_identity_sha256=self.engine.plan.identity_sha256,
                dtype=self.engine.dtype,
                device=self.engine.device,
                batch=self.engine.batch,
            )
        else:
            restore_periodic_state(
                self.engine, self.state, load_periodic_checkpoint(path), run_spec=self.spec
            )
        self.index = step

    def identity(self):
        if not self.engine_periodic:
            return super().identity()
        stokes = self.spec.simulation.equation_system.parameters["stokes"]["parameters"]
        policy = stokes["tangential_zero_mode_policy"]
        return {
            "configuration": self.configuration_identity,
            "runtime": {
                "requested": "periodic_spectral",
                "effective": "periodic_spectral",
                "adapter": "Engine",
                "fallback_used": False,
                "pressure_gauge": "zero_mean",
                "hermitian_state_projection": self.engine.projector.execution_metadata()[
                    "hermitian_state_projection"
                ],
                "tangential_zero_mode_policy": policy,
                "friction": stokes["friction"],
                "uniform_velocity_mode_action": "remove_all_uniform_velocity_and_force"
                if policy == "zero_mean"
                else "retain_all_uniform_velocity_resolved_by_friction",
            },
        }
