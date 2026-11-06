"""Read-only qualification adapters for the v0.4 production and functional APIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.models.active_nematics import Q_COMPONENTS, CompleteStressBerisEdwards
from qualification.identity.records import tensor_record


def initial_values(scene: dict) -> np.ndarray:
    """Smooth, deterministic perturbation; no random backend-dependent generators."""
    shape = tuple(scene["shape"])
    grid = np.indices(shape, dtype=np.float64)
    wave = sum(np.cos(2 * np.pi * grid[i] / shape[i]) for i in range(3))
    values = np.stack([(i + 1) * 1e-4 * wave for i in range(5)], axis=-1)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    return values


def complete_simulation(scene: dict, root: Path) -> Simulation:
    from pssolver.geometries import PeriodicBox, RectangularChannel

    channel = scene["family"].endswith("channel")
    options = scene["options"]
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1 / 81,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2 / 3,
    )
    geometry_type = RectangularChannel if channel else PeriodicBox
    geometry = geometry_type(shape=tuple(scene["shape"]), lengths=(16.0, 12.0, 12.0))
    snapshot = root / "initial"
    snapshot.mkdir(exist_ok=True)
    np.save(snapshot / "Q_0.npy", initial_values(scene), allow_pickle=False)
    pressure = (
        {
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1e-10,
                "max_iterations": 40,
                "fixed_iterations": options.get("fixed_iterations", 12),
                "warm_start": options.get("warm_start", True),
            }
        }
        if channel
        else {}
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": no_slip_velocity() if channel else free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype=scene["dtype"],
            dealias_rule=options.get("dealias_rule", "cubic_half"),
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress" if channel else "periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical" if channel else "spectral",
                "pointwise_execution": options.get("pointwise_execution", "eager"),
                "disable_q_gradient_reuse": options.get("disable_q_gradient_reuse", False),
            },
        ),
        output=Output(
            directory=root / "output",
            steps=scene["steps"],
            save_interval=scene["steps"],
            diagnostic_interval=scene["steps"],
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=pressure,
    )


class RuntimeCase:
    """Keep construction, stepping, observation and checkpoint side effects explicit."""

    def __init__(self, scene: dict, root: Path):
        self.scene = scene
        self.functional = scene["family"].startswith("functional_")
        self.family = scene["family"].removeprefix("functional_")
        self.index = 0
        self.state = None
        if self.family == "plane":
            self._plane(root)
        else:
            simulation = complete_simulation(scene, root)
            self.configuration_identity = simulation.identity_metadata()
            self._complete(simulation)
        # Functional API 1.1 exposes a stable wrapper, not its private solver.
        self.solver = None if self.functional else self.runtime.solver

    def _plane(self, root: Path) -> None:
        from pssolver.configuration import create_plane_beris_edwards_run_spec
        from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
        from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
        from pssolver.workflows.plane_compiled_v2 import build_plane_compiled_v2_runtime

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
            pointwise_execution="eager",
            tf32="off",
            runtime_path=self.scene["runtime_path"],
            **options,
        )
        values = initial_values(self.scene)
        initial = {
            name: torch.tensor(values[..., i], dtype=getattr(torch, self.scene["dtype"]))
            for i, name in enumerate(Q_COMPONENTS)
        }
        if self.scene["runtime_path"] == "compiled_v2":
            self.runtime = build_plane_compiled_v2_runtime(
                self.spec, device="cpu", initial_values=initial
            )
        else:
            solver, projector = build_legacy_plane_runtime(
                self.spec, device="cpu", initial_values=initial
            )
            self.runtime = LegacyPlaneRuntimeAdapter(solver, projector)
        self.configuration_identity = self.spec.identity_metadata()

    def _complete(self, simulation: Simulation) -> None:
        channel = self.family == "channel"
        if self.functional:
            from pssolver.functional.api import (
                build_channel_activity_functional_runtime,
                build_functional_runtime,
                channel_activity_functional_request,
                periodic_activity_functional_request,
            )

            request = (
                channel_activity_functional_request
                if channel
                else periodic_activity_functional_request
            )(simulation.specification)
            self.runtime = (
                build_channel_activity_functional_runtime if channel else build_functional_runtime
            )(request)
            self.state = self.runtime.initial_state()
            spec = self.runtime.control_specs[0].tensor
            self.controls = {
                "activity": torch.full(spec.shape, 0.01, dtype=self.state[0].dtype, device="cpu")
            }
            return
        self.spec = compile_simulation(simulation).application_request
        if channel:
            from pssolver.applications.channel_beris_edwards import load_channel_initial_q
            from pssolver.runtime.channel_beris_edwards import (
                ChannelBerisEdwardsRuntimeBuildRequest,
                build_channel_beris_edwards_runtime,
            )

            values, _, _ = load_channel_initial_q(self.spec)
            self.runtime = build_channel_beris_edwards_runtime(
                ChannelBerisEdwardsRuntimeBuildRequest(self.spec, values, "cpu")
            )
        else:
            from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
            from pssolver.runtime.periodic_beris_edwards import (
                PeriodicRuntimeBuildRequest,
                build_periodic_beris_edwards_runtime,
            )

            values, _, _ = load_periodic_initial_q(self.spec)
            self.runtime = build_periodic_beris_edwards_runtime(
                PeriodicRuntimeBuildRequest(self.spec, values, "cpu")
            )

    def step(self) -> None:
        if self.functional:
            self.state = self.runtime.step(self.state, self.controls, self.index)
        else:
            self.runtime.advance(1)
        self.index += 1

    def observe(self) -> dict:
        if self.functional:
            return {
                name: tensor_record(value)
                for name, value in self.runtime.observe(self.state, self.controls).items()
            }
        self.runtime.synchronize_for_observation()
        return {
            name: tensor_record(self.runtime.fields[name])
            for name in self.runtime.fields.name_to_idx
        }

    def record(self) -> dict:
        if self.functional:
            return {
                "step": self.index,
                "tensors": {
                    f"state_{i}": tensor_record(value) for i, value in enumerate(self.state)
                },
            }
        fields = self.runtime.fields
        integrator = self.solver.integrator
        tensors = {
            name + suffix: tensor_record(fields[name + suffix])
            for name in fields.name_to_idx
            for suffix in ("", ".hat")
        }
        flow = self.solver.model.static_model
        guess = getattr(flow, "pressure_guess", None)
        if isinstance(guess, torch.Tensor):
            tensors["pressure_guess"] = tensor_record(guess)
        return {
            "step": self.index,
            "tensors": tensors,
            "progress": {
                "step_count": int(integrator.step_count),
                "refresh_count": int(integrator.refresh_count),
                "refresh_interval": integrator.spectral_refresh_interval,
                "constraints_current": bool(integrator._static_fields_are_current),
            },
        }

    def export(self, path: Path) -> None:
        if self.functional:
            self.runtime.checkpoint_bridge.export_checkpoint(
                path, self.state, completed_steps=self.index
            )
        elif self.family == "plane":
            from pssolver.workflows.plane_checkpoint import (
                capture_plane_checkpoint,
                write_plane_checkpoint,
            )

            write_plane_checkpoint(path, capture_plane_checkpoint(self.runtime, run_spec=self.spec))
        elif self.family == "periodic":
            from pssolver.workflows.periodic_checkpoint import (
                capture_periodic_checkpoint,
                write_periodic_checkpoint,
            )

            write_periodic_checkpoint(
                path, capture_periodic_checkpoint(self.runtime, run_spec=self.spec)
            )
        else:
            from pssolver.workflows.channel_beris_edwards import (
                write_channel_beris_edwards_checkpoint,
            )

            write_channel_beris_edwards_checkpoint(path, self.runtime, run_spec=self.spec)

    def restore(self, path: Path, step: int) -> None:
        if self.functional:
            self.state = self.runtime.checkpoint_bridge.import_checkpoint(path).state
        elif self.family == "plane":
            from pssolver.workflows.plane_checkpoint import (
                load_plane_checkpoint,
                restore_plane_checkpoint,
            )

            restore_plane_checkpoint(self.runtime, load_plane_checkpoint(path), run_spec=self.spec)
        elif self.family == "periodic":
            from pssolver.workflows.periodic_checkpoint import (
                load_periodic_checkpoint,
                restore_periodic_checkpoint,
            )

            restore_periodic_checkpoint(
                self.runtime, load_periodic_checkpoint(path), run_spec=self.spec
            )
        else:
            from pssolver.workflows.channel_beris_edwards import _load_checkpoint

            _load_checkpoint(path, self.runtime, run_spec=self.spec)
        self.index = step

    def identity(self) -> dict:
        runtime = (
            self.runtime.identity().to_metadata() if self.functional else self.runtime.to_metadata()
        )
        return {"configuration": self.configuration_identity, "runtime": runtime}
