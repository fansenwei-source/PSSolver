"""Production Engine test driver; the frozen v0.4 v1 driver stays intact."""

import json
from pathlib import Path

import torch

from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import RuntimeCase, complete_simulation, initial_values


class RuntimeCaseV2(RuntimeCase):
    """Use Engine/State for production; retain the stable functional driver."""

    def __init__(self, scene: dict, root: Path):
        self._channel = None
        self.engine_family = scene["family"] in {"periodic", "plane", "channel"}
        if scene["family"] == "channel":
            self._engine_channel(scene, root)
            return
        if scene["family"] == "plane":
            self._engine_plane(scene, root)
            return
        if not self.engine_family:
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

    def _engine_channel(self, scene, root):
        from qualification.identity.channel_comparison import ChannelEngineCase

        self.scene, self.family, self.functional = scene, "channel", False
        self._channel = ChannelEngineCase(scene, root)

    def _engine_plane(self, scene, root):
        self.scene, self.family, self.functional = scene, "plane", False
        self.index = 0
        from pssolver.configuration import create_plane_beris_edwards_run_spec
        from pssolver.configuration.legacy_plane import translate_plane
        from pssolver.engine import Engine
        from pssolver.models.active_nematics import Q_COMPONENTS

        options = dict(self.scene["options"])
        options.pop("shape", None)
        self.spec = create_plane_beris_edwards_run_spec(
            activity_number=18.0,
            output_dir=root / "output",
            nx=self.scene["shape"][0],
            ny=self.scene["shape"][1],
            nz=self.scene["shape"][2],
            lx=16.0,
            ly=16.0,
            height=12.0,
            parameterization="fixed-k",
            frank_k=1 / 81,
            dt=0.001,
            steps=self.scene["steps"],
            save_start_step=0,
            save_interval=64,
            diagnostic_interval=64,
            device="cpu",
            dtype=self.scene["dtype"],
            pointwise_execution=options.pop("pointwise_execution", "eager"),
            tf32="off",
            runtime_path=self.scene["runtime_path"],
            **options,
        )
        values = initial_values(self.scene)
        initial = {
            name: torch.tensor(values[..., i], dtype=getattr(torch, self.scene["dtype"]))
            for i, name in enumerate(Q_COMPONENTS)
        }
        self.configuration_identity = self.spec.identity_metadata()
        self.engine = Engine(
            translate_plane(self.spec), device="cpu", dtype=getattr(torch, scene["dtype"])
        )
        self.last_diagnostics = {}
        self.state = self.engine.initial_state(initial, diagnostics=self.last_diagnostics)

    def prepare(self):
        if self._channel is not None:
            return self._channel.prepare()
        self.engine.prepare_constraints_(
            self.state,
            controls={"alpha": self.engine.default_controls["alpha"]},
            diagnostics=self.last_diagnostics,
        )

    def step(self):
        if self._channel is not None:
            return self._channel.step()
        if not self.engine_family:
            return super().step()
        self.prepare()
        before = self.record()
        self.engine.step_(self.state, diagnostics=self.last_diagnostics)
        self.index += 1
        return before

    def observe(self):
        if self._channel is not None:
            return self._channel.observe()
        if not self.engine_family:
            return super().observe()
        self.prepare()
        fields = self.engine.field_view(self.state)
        return {name: tensor_record(fields[name]) for name in self.engine.components}

    def record(self):
        if self._channel is not None:
            return self._channel.record()
        if not self.engine_family:
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
                "pressure_iterations": self.last_diagnostics[
                    "last_pressure_iterations" if self.family == "plane" else "pressure_iterations"
                ],
            },
        }

    def export(self, path):
        if self._channel is not None:
            return self._channel.export(path)
        if not self.engine_family:
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
        if self._channel is not None:
            return self._channel.restore(path, step)
        if not self.engine_family:
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
        elif self.family == "plane":
            from pssolver.configuration.legacy_plane import load_plane_checkpoint
            from pssolver.runtime.legacy_entry import restore_plane_state

            restore_plane_state(
                self.engine,
                self.state,
                load_plane_checkpoint(path),
                run_spec=self.spec,
                diagnostics=self.last_diagnostics,
            )
        else:
            restore_periodic_state(
                self.engine, self.state, load_periodic_checkpoint(path), run_spec=self.spec
            )
        self.index = step

    def identity(self):
        if self._channel is not None:
            return self._channel.identity()
        if not self.engine_family:
            return super().identity()
        if self.family == "plane":
            return {
                "configuration": self.configuration_identity,
                "runtime": {
                    "requested": self.spec.runtime_path.value,
                    "effective": self.spec.runtime_path.value,
                    "adapter": "Engine",
                    "fallback_used": False,
                    "separated_architecture": None,
                },
            }
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
