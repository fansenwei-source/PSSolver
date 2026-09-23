"""Equation-level contracts for the canonical no-slip Channel Stokes solve."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.channel import ModalSaddleStokesCompute
from pssolver.experimental import stokes as experimental_stokes
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)


SHAPE = (7, 6, 5)
LENGTHS = (4.0, 3.0, 2.5)
FRICTION = 0.2
VISCOSITY = 0.73
ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _spectral_solver(shape=SHAPE):
    lengths = LENGTHS if shape == SHAPE else tuple(float(n) for n in shape)
    return SpectralSolver(
        shape,
        L=lengths,
        device="cpu",
        dtype=torch.float64,
    )


def _canonical(solver, **overrides):
    options = {
        "friction": FRICTION,
        "viscosity": VISCOSITY,
        "pressure_relative_tolerance": 1.0e-12,
        "pressure_max_iterations": 300,
    }
    options.update(overrides)
    return ChannelNoSlipModalStokesSolver(
        solver.transform_backend,
        **options,
    )


def _force_hats(solver, *, batch_size=2):
    generator = torch.Generator().manual_seed(472)
    forces = tuple(
        torch.randn(
            (batch_size, *SHAPE),
            generator=generator,
            dtype=torch.float64,
        )
        for _ in range(3)
    )
    return tuple(
        solver.transform_tensor(
            force,
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
        )
        for force in forces
    )


def _relative_norm(value, reference):
    return float(
        torch.linalg.vector_norm(value.reshape(-1))
        / torch.linalg.vector_norm(reference.reshape(-1))
    )


def test_channel_boundary_spaces_and_registered_operator_state_are_frozen():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    velocity_metadata = solver.transform_backend.get_metadata(
        CHANNEL_VELOCITY_BOUNDARY_CONDITIONS
    )
    pressure_metadata = solver.transform_backend.get_metadata(
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS
    )

    assert CHANNEL_VELOCITY_BOUNDARY_CONDITIONS == (
        "periodic",
        "dirichlet",
        "dirichlet",
    )
    assert CHANNEL_PRESSURE_BOUNDARY_CONDITIONS == (
        "periodic",
        "neumann",
        "neumann",
    )
    assert velocity_metadata.transform_kinds == ("fft", "dst", "dst")
    assert pressure_metadata.transform_kinds == ("fft", "dct", "dct")
    assert tuple(stokes.state_dict()) == (
        "ikx",
        "a_inv",
        "n_to_d_y",
        "d_to_n_y",
        "dn_to_dd_y",
        "dd_to_dn_y",
        "n_to_d_z",
        "d_to_n_z",
        "dn_to_dd_z",
        "dd_to_dn_z",
        "schur_diag_safe",
        "pressure_null_mask",
    )


def test_legacy_active_force_facade_preserves_canonical_linear_solve_exactly():
    solver = _spectral_solver()
    canonical = _canonical(solver)
    legacy = ModalSaddleStokesCompute(
        solver,
        beta=-1.0,
        friction=FRICTION,
        viscosity=VISCOSITY,
        pressure_rel_tol=1.0e-12,
        pressure_max_iter=300,
    )

    assert tuple(legacy.state_dict()) == tuple(canonical.state_dict())
    for name, value in canonical.state_dict().items():
        torch.testing.assert_close(
            legacy.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    force_hats = _force_hats(solver)
    expected = canonical.solve_force_hats(*force_hats)
    actual = legacy.solve_force_hats(*force_hats)
    for actual_hat, expected_hat in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_hat,
            expected_hat,
            rtol=0.0,
            atol=0.0,
        )
    assert legacy.last_pressure_iterations == canonical.last_pressure_iterations
    assert legacy.last_pressure_residual == canonical.last_pressure_residual


def test_force_solve_satisfies_momentum_incompressibility_and_pressure_gauge():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    force_hats = _force_hats(solver)

    ux_hat, uy_hat, uz_hat, pressure_hat = stokes.solve_force_hats(
        *force_hats
    )
    velocity_hats = (ux_hat, uy_hat, uz_hat)
    pressure_gradient_hats = stokes.pressure_gradient_hats(pressure_hat)

    for force_hat, velocity_hat, gradient_hat in zip(
        force_hats,
        velocity_hats,
        pressure_gradient_hats,
        strict=True,
    ):
        momentum_residual = (
            force_hat - gradient_hat - velocity_hat / stokes.a_inv
        )
        assert _relative_norm(momentum_residual, force_hat) < 2.0e-15

    divergence_hat = stokes.divergence_hat(*velocity_hats)
    velocity_norm = sum(
        torch.linalg.vector_norm(value.reshape(-1))
        for value in velocity_hats
    )
    assert float(
        torch.linalg.vector_norm(divergence_hat.reshape(-1)) / velocity_norm
    ) < 2.0e-11
    assert bool((pressure_hat[:, 0, 0, 0] == 0).all().item())
    assert stokes.last_pressure_relative_residual < 2.0e-12


def test_manufactured_schur_problem_recovers_the_zero_gauge_pressure():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    generator = torch.Generator().manual_seed(813)
    physical_pressure = torch.randn(
        (2, *SHAPE),
        generator=generator,
        dtype=torch.float64,
    )
    exact_pressure_hat = solver.transform_tensor(
        physical_pressure,
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    )
    exact_pressure_hat = stokes._project_pressure_gauge(exact_pressure_hat)
    rhs_hat = stokes._pressure_operator(exact_pressure_hat)

    actual_pressure_hat = stokes._solve_pressure(rhs_hat)

    assert _relative_norm(
        stokes._pressure_operator(actual_pressure_hat) - rhs_hat,
        rhs_hat,
    ) < 2.0e-12
    assert _relative_norm(
        actual_pressure_hat - exact_pressure_hat,
        exact_pressure_hat,
    ) < 3.0e-12
    assert bool((actual_pressure_hat[:, 0, 0, 0] == 0).all().item())


def test_zero_force_has_exact_zero_solution_and_diagnostics():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    zero = torch.zeros(
        (2, *solver.spectral_shape),
        dtype=torch.complex128,
    )

    solution = stokes.solve_force_hats(zero, zero, zero)

    assert all(bool((value == 0).all().item()) for value in solution)
    assert stokes.last_pressure_iterations == 0
    assert stokes.last_pressure_residual == 0.0
    assert stokes.last_pressure_relative_residual == 0.0


def test_fixed_iteration_pressure_mode_preserves_the_requested_work_count():
    solver = _spectral_solver()
    stokes = _canonical(
        solver,
        pressure_relative_tolerance=1.0e-30,
        pressure_fixed_iterations=4,
    )

    stokes.solve_force_hats(*_force_hats(solver))

    assert stokes.last_pressure_iterations == 4
    assert stokes.pressure_fixed_iterations == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"friction": -0.1}, "friction"),
        ({"viscosity": 0.0}, "viscosity"),
        ({"pressure_relative_tolerance": 0.0}, "relative_tolerance"),
        ({"pressure_max_iterations": 0}, "max_iterations"),
        ({"pressure_fixed_iterations": 0}, "fixed_iterations"),
    ),
)
def test_constructor_rejects_invalid_solver_coefficients(overrides, message):
    with pytest.raises(ValueError, match=message):
        _canonical(_spectral_solver(), **overrides)


def test_constructor_rejects_non_three_dimensional_transform_backend():
    solver = _spectral_solver(shape=(7, 6))

    with pytest.raises(ValueError, match="three dimensions"):
        _canonical(solver)


def test_canonical_solver_is_model_independent_and_adapter_uses_it_directly():
    canonical_source = inspect.getsource(ChannelNoSlipModalStokesSolver)
    adapter_source = inspect.getsource(experimental_stokes)

    assert "active_force" not in canonical_source
    assert "Q_COMPONENTS" not in canonical_source
    assert "from pssolver.channel import" not in adapter_source
    assert "ChannelNoSlipModalStokesSolver(" in adapter_source


def test_p72_record_binds_the_extracted_solver_and_limits_authorization():
    value = json.loads(
        (NOTES / "phase_7_p72_channel_stokes_extraction.json").read_text(
            encoding="utf-8"
        )
    )

    def sha256(relative):
        return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    assert value["classification"] == (
        "PASS_P7_2_CHANNEL_STOKES_SCHUR_EXTRACTION"
    )
    assert sha256("Channel.py") == value["compatibility"][
        "channel_entry_point_sha256"
    ]
    assert sha256("pssolver/channel.py") == value["compatibility"][
        "legacy_facade_module_sha256"
    ]
    assert sha256("pssolver/experimental/stokes.py") == value[
        "compatibility"
    ]["experimental_adapter_module_sha256"]
    assert sha256(
        "pssolver/linear_solvers/stokes/channel_no_slip.py"
    ) == value["canonical_solver"]["module_sha256"]
    assert value["authorization"] == {
        "p7_2_complete": True,
        "p7_3_planning_eligible": True,
        "p7_3_implementation_authorized": False,
        "phase_7_h100_authorized": False,
        "new_boundary_law_authorized": False,
        "production_default_changed": False,
        "plane_evidence_modified": False,
    }
