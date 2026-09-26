"""Projected divergence operators for three-dimensional row-major tensors.

The qualified distortion operator uses axis 2 as the distinguished wall-normal
direction and preserves the frozen even/odd basis split.  This module does not
construct model stress and does not claim a generic N-dimensional contract.
"""

import torch

from .projection import _forward_projected, _inverse_projected


def _inverse_spectral_gradient(
    backend,
    spectral,
    boundary_conditions,
    axis,
    *,
    projector=None,
):
    gradient_hat, gradient_bcs = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis,
    )
    return _inverse_projected(
        backend,
        projector,
        gradient_hat,
        gradient_bcs,
    )


def _validate_divergence_sum_space(sum_space):
    if sum_space not in {"physical", "spectral"}:
        raise ValueError(
            "sum_space must be 'physical' or 'spectral'."
        )


def _spectral_gradient(backend, spectral, boundary_conditions, axis):
    return backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis,
    )


def projected_common_basis_stress_divergence(
    backend,
    stress_components,
    boundary_conditions,
    *,
    projector=None,
    sum_space="physical",
):
    """Return row-wise ``partial_j stress_ij`` for one shared basis.

    Components are the nine row-major entries
    ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)``.
    ``sum_space='spectral'`` combines the x/y derivative coefficients, which
    share a basis, before inversion.  The z derivative retains its parity-
    changed basis and is combined in physical space.
    """
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    _validate_divergence_sum_space(sum_space)
    boundary_conditions = tuple(boundary_conditions)
    stress_hat = _forward_projected(
        backend,
        projector,
        torch.stack(tuple(stress_components)),
        boundary_conditions,
    )
    stress_matrix = stress_hat.unflatten(0, (3, 3))
    if sum_space == "physical":
        derivative_x = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 0],
            boundary_conditions,
            axis=0,
            projector=projector,
        )
        derivative_y = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 1],
            boundary_conditions,
            axis=1,
            projector=projector,
        )
        derivative_z = _inverse_spectral_gradient(
            backend,
            stress_matrix[:, 2],
            boundary_conditions,
            axis=2,
            projector=projector,
        )
        return derivative_x + derivative_y + derivative_z

    derivative_x_hat, derivative_x_bcs = _spectral_gradient(
        backend,
        stress_matrix[:, 0],
        boundary_conditions,
        axis=0,
    )
    derivative_y_hat, derivative_y_bcs = _spectral_gradient(
        backend,
        stress_matrix[:, 1],
        boundary_conditions,
        axis=1,
    )
    if derivative_x_bcs != derivative_y_bcs:
        raise RuntimeError(
            "The x/y stress derivatives must share one spectral basis."
        )
    derivative_xy = _inverse_projected(
        backend,
        projector,
        derivative_x_hat + derivative_y_hat,
        derivative_x_bcs,
    )
    derivative_z = _inverse_spectral_gradient(
        backend,
        stress_matrix[:, 2],
        boundary_conditions,
        axis=2,
        projector=projector,
    )
    return derivative_xy + derivative_z


def projected_distortion_stress_divergence(
    backend,
    stress_components,
    even_boundary_conditions,
    odd_boundary_conditions,
    *,
    projector=None,
    sum_space="physical",
):
    """Differentiate row-major distortion stress with its z parity split.

    ``sum_space='spectral'`` assembles the two tangential components in their
    common Neumann basis and the normal component in its Dirichlet basis before
    inversion.  This is the same linear divergence with fewer transforms.
    """
    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    _validate_divergence_sum_space(sum_space)
    even_boundary_conditions = tuple(even_boundary_conditions)
    odd_boundary_conditions = tuple(odd_boundary_conditions)
    even_components = torch.stack(
        tuple(stress_components[index] for index in (0, 1, 3, 4, 8))
    )
    odd_components = torch.stack(
        tuple(stress_components[index] for index in (2, 5, 6, 7))
    )
    even_hat = _forward_projected(
        backend,
        projector,
        even_components,
        even_boundary_conditions,
    )
    odd_hat = _forward_projected(
        backend,
        projector,
        odd_components,
        odd_boundary_conditions,
    )
    even_matrix = even_hat[:4].unflatten(0, (2, 2))
    odd_tangential = odd_hat[:2]

    if sum_space == "physical":
        even_x = _inverse_spectral_gradient(
            backend,
            even_matrix[:, 0],
            even_boundary_conditions,
            axis=0,
            projector=projector,
        )
        even_y = _inverse_spectral_gradient(
            backend,
            even_matrix[:, 1],
            even_boundary_conditions,
            axis=1,
            projector=projector,
        )
        even_z = _inverse_spectral_gradient(
            backend,
            even_hat[4],
            even_boundary_conditions,
            axis=2,
            projector=projector,
        )
        odd_x = _inverse_spectral_gradient(
            backend,
            odd_hat[2],
            odd_boundary_conditions,
            axis=0,
            projector=projector,
        )
        odd_y = _inverse_spectral_gradient(
            backend,
            odd_hat[3],
            odd_boundary_conditions,
            axis=1,
            projector=projector,
        )
        odd_z = _inverse_spectral_gradient(
            backend,
            odd_tangential,
            odd_boundary_conditions,
            axis=2,
            projector=projector,
        )
        return torch.stack(
            (
                even_x[0] + even_y[0] + odd_z[0],
                even_x[1] + even_y[1] + odd_z[1],
                odd_x + odd_y + even_z,
            )
        )

    even_x_hat, even_x_bcs = _spectral_gradient(
        backend, even_matrix[:, 0], even_boundary_conditions, axis=0
    )
    even_y_hat, even_y_bcs = _spectral_gradient(
        backend, even_matrix[:, 1], even_boundary_conditions, axis=1
    )
    odd_z_hat, odd_z_bcs = _spectral_gradient(
        backend, odd_tangential, odd_boundary_conditions, axis=2
    )
    if not (even_x_bcs == even_y_bcs == odd_z_bcs):
        raise RuntimeError(
            "Tangential distortion-force terms must share one spectral basis."
        )
    tangential = _inverse_projected(
        backend,
        projector,
        even_x_hat + even_y_hat + odd_z_hat,
        even_x_bcs,
    )

    odd_x_hat, odd_x_bcs = _spectral_gradient(
        backend, odd_hat[2], odd_boundary_conditions, axis=0
    )
    odd_y_hat, odd_y_bcs = _spectral_gradient(
        backend, odd_hat[3], odd_boundary_conditions, axis=1
    )
    even_z_hat, even_z_bcs = _spectral_gradient(
        backend, even_hat[4], even_boundary_conditions, axis=2
    )
    if not (odd_x_bcs == odd_y_bcs == even_z_bcs):
        raise RuntimeError(
            "Normal distortion-force terms must share one spectral basis."
        )
    normal = _inverse_projected(
        backend,
        projector,
        odd_x_hat + odd_y_hat + even_z_hat,
        odd_x_bcs,
    )
    return torch.stack((tangential[0], tangential[1], normal))


def projected_component_basis_stress_divergence(
    backend,
    stress_components,
    component_boundary_conditions,
    *,
    output_boundary_conditions,
    projector=None,
):
    """Return a row-wise stress divergence with per-component bases.

    This is the fail-closed two-bounded-axis path used by the rectangular
    Channel complete-stress model.  Each row-major stress component declares
    its native tensor-product basis.  Derivatives are evaluated in that basis,
    summed in physical space, and the three resulting force components are
    projected together into the native velocity basis.

    The physical-space sum is intentional: with two bounded directions the
    three derivatives in one stress row generally occupy three different
    intermediate parity spaces.  Treating them as a common spectral basis
    would be mathematically invalid.
    """

    if len(stress_components) != 9:
        raise ValueError("A three-dimensional stress requires nine components.")
    component_boundary_conditions = tuple(
        tuple(value) for value in component_boundary_conditions
    )
    if len(component_boundary_conditions) != 9:
        raise ValueError(
            "component_boundary_conditions must contain nine basis tuples."
        )
    ndim = len(tuple(output_boundary_conditions))
    if ndim != 3 or any(
        len(value) != ndim for value in component_boundary_conditions
    ):
        raise ValueError("stress and output bases must be three-dimensional.")

    rows = []
    for row in range(3):
        terms = []
        for axis in range(3):
            index = 3 * row + axis
            boundary_conditions = component_boundary_conditions[index]
            spectral = _forward_projected(
                backend,
                projector,
                stress_components[index],
                boundary_conditions,
            )
            terms.append(
                _inverse_spectral_gradient(
                    backend,
                    spectral,
                    boundary_conditions,
                    axis,
                    projector=projector,
                )
            )
        rows.append(sum(terms))

    physical_force = torch.stack(tuple(rows))
    return _inverse_projected(
        backend,
        projector,
        _forward_projected(
            backend,
            projector,
            physical_force,
            tuple(output_boundary_conditions),
        ),
        tuple(output_boundary_conditions),
    )
