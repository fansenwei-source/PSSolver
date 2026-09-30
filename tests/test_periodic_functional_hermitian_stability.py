"""Long-horizon regression for packed-spectrum periodic functional state."""

from __future__ import annotations

import numpy as np
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.functional.api import (
    build_functional_runtime,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.models.active_nematics.q_tensor import positive_equilibrium_S


SHAPE = (32, 32, 8)
LENGTHS = (32.0, 32.0, 8.0)
DT = 0.02
HORIZON = 200.0


def _uniform_q(order):
    director = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    tensor = 1.5 * order * (
        np.outer(director, director) - np.eye(3) / 3.0
    )
    return np.array(
        (tensor[0, 0], tensor[0, 1], tensor[0, 2], tensor[1, 1], tensor[1, 2])
    )


def _packed_plane_violation(spectral):
    ny = SHAPE[1]
    plane_indices = torch.tensor(
        (0, ny // 2),
        device=spectral.device,
    )
    planes = spectral.index_select(-2, plane_indices)
    reverse_x = torch.remainder(
        -torch.arange(SHAPE[0], device=spectral.device),
        SHAPE[0],
    )
    reverse_z = torch.remainder(
        -torch.arange(SHAPE[2], device=spectral.device),
        SHAPE[2],
    )
    reflected = planes.index_select(-3, reverse_x).index_select(-1, reverse_z)
    return (planes - reflected.conj()).abs().max()


def _runtime(tmp_path):
    material = {
        "ldg_a": -1.0,
        "ldg_b": -6.0,
        "ldg_c": 6.0,
        "ldg_l1": 1.0,
        "gamma": 1.0,
        "flow_alignment": 1.0,
        "viscosity": 1.0,
        "beta": -1.0,
        "friction": 0.0,
        "tangential_zero_mode_policy": "zero_mean",
    }
    order = float(
        positive_equilibrium_S(
            material["ldg_a"],
            material["ldg_b"],
            material["ldg_c"],
        )
    )
    q = np.broadcast_to(
        _uniform_q(order)[:, None, None, None],
        (5, *SHAPE),
    ).copy()
    q += np.random.default_rng(0).normal(scale=1.0e-3, size=q.shape)
    np.save(tmp_path / "Q_0.npy", q, allow_pickle=False)

    model = CompleteStressBerisEdwards(
        **material,
        activity=0.05,
    )
    geometry = PeriodicBox(shape=SHAPE, lengths=LENGTHS)
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=DT, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(tmp_path, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=tmp_path / "output",
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
        ),
    )
    return build_functional_runtime(
        periodic_activity_functional_request(simulation.specification)
    )


def test_periodic_functional_state_remains_hermitian_through_time_200(tmp_path):
    """Prevent hidden anti-Hermitian growth when refresh is disabled.

    The unfixed runtime becomes non-finite near ``t=86.48`` for this case,
    while its physical inverse remains deceptively regular until the hidden
    packed-plane component leaks into the retained dynamics.
    """

    runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    control_spec = runtime.control_specs[0]
    controls = {
        "activity": torch.full(
            control_spec.tensor.shape,
            0.05,
            device=state[0].device,
            dtype=torch.float64,
        )
    }
    projection = runtime.identity().to_metadata()["execution"][
        "functional_runtime"
    ]["hermitian_state_projection"]
    assert projection == "self_conjugate_planes_each_step"

    steps = round(HORIZON / DT)
    with torch.no_grad():
        for step in range(steps):
            state = runtime.step(state, controls, step)
            if (step + 1) % 250 == 0:
                assert bool(torch.isfinite(state[0]).all())
                assert bool(torch.isfinite(state[1]).all())
                assert _packed_plane_violation(state[1]).item() == 0.0

    assert (step + 1) * DT == HORIZON
    assert bool(torch.isfinite(state[0]).all())
    assert bool(torch.isfinite(state[1]).all())
    assert _packed_plane_violation(state[1]).item() == 0.0
