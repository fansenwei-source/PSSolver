import pytest
import torch

from pssolver import SpectralSolver
from pssolver.Field import Fields
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    aligned_x_smooth_noise,
    analytic_periodic_defect_gas_2d,
    extruded_2d_twist,
    neumann_twist_profile,
)
from pssolver.transforms import TensorProductTransformBackend


@pytest.mark.parametrize(
    ("dtype", "spectral_dtype"),
    ((torch.float32, torch.complex64), (torch.float64, torch.complex128)),
)
def test_solver_build_preserves_requested_dtype(dtype, spectral_dtype):
    shape = (5, 4, 3)
    solver = SpectralSolver(
        shape,
        L=(5.0, 4.0, 3.0),
        device="cpu",
        dtype=dtype,
    )
    solver.model.add_dynamic_field(
        "q",
        init=torch.ones(shape, dtype=torch.float32),
        L_hat=-solver.get_q2(("periodic", "periodic", "neumann")),
        boundary_conditions=("periodic", "periodic", "neumann"),
    )
    solver.model.add_static_field(
        "u",
        boundary_conditions=("periodic", "periodic", "neumann"),
    )
    solver.build()

    assert solver.dtype == dtype
    assert solver.transform_backend.real_dtype == dtype
    assert solver.fields.dtype == dtype
    assert solver.fields.spatial.dtype == dtype
    assert solver.fields.L_hat.dtype == dtype
    assert solver.fields.spectral.dtype == spectral_dtype
    assert solver.fields["u"].dtype == dtype
    assert solver.qx.dtype == dtype
    assert solver.q2.dtype == dtype


def test_fields_reject_transform_backend_with_mismatched_dtype():
    fields = Fields((4,), device="cpu", dtype=torch.float64)
    backend = TensorProductTransformBackend(
        (4,),
        (1.0,),
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="must match"):
        fields.set_transform_backend(backend)


@pytest.mark.parametrize("invalid_dtype", (torch.float16, torch.complex64))
def test_solver_rejects_unsupported_dtype(invalid_dtype):
    with pytest.raises(ValueError, match="torch.float32 or torch.float64"):
        SpectralSolver((4,), device="cpu", dtype=invalid_dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_active_nematic_initial_conditions_preserve_requested_dtype(dtype):
    defect_fields = analytic_periodic_defect_gas_2d(
        (12, 10),
        lengths=(12.0, 10.0),
        num_defect_pairs=1,
        min_separation=2.0,
        core_radius=0.8,
        S_initial=0.4,
        seed=7,
        dtype=dtype,
    )
    twist = neumann_twist_profile(7, seed=3, dtype=dtype)
    extruded_fields = extruded_2d_twist(
        (12, 10, 7),
        Q_2d=defect_fields,
        boundary_conditions=("periodic", "periodic", "neumann"),
        seed=3,
        dtype=dtype,
    )
    noisy_fields = aligned_x_smooth_noise(
        (5, 4, 3),
        boundary_conditions=("periodic", "periodic", "neumann"),
        S_initial=0.4,
        seed=5,
        dtype=dtype,
    )

    assert twist.dtype == dtype
    for fields in (defect_fields, extruded_fields, noisy_fields):
        assert tuple(fields) == Q_COMPONENTS
        assert all(values.dtype == dtype for values in fields.values())
