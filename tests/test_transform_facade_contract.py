"""Frozen compatibility contract for the v0.1.2 transform facade.

Phase 1 changes implementation ownership, not the historical imports,
signatures, constants, or checkpoint format.  Keep these assertions separate
from numerical reference tests so a mechanical move cannot silently narrow
the compatibility surface.
"""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import pssolver
import pssolver.transforms as transforms
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_FACADE_NAMES = (
    "DEFAULT_DEALIAS_RULE",
    "DEFAULT_TRANSFORM_EXECUTION_ORDER",
    "DEFAULT_PROJECTED_TRANSFORM_EXECUTION",
    "PROJECTED_TRANSFORM_EXECUTION_MODES",
    "DEFAULT_SPECTRAL_STORAGE",
    "SPECTRAL_STORAGE_MODES",
    "DEFAULT_PERIODIC_TRANSFORM_EXECUTION",
    "PERIODIC_TRANSFORM_EXECUTION_MODES",
    "DEALIAS_RULE_FRACTIONS",
    "TransformMetadata",
    "TensorProductTransformBackend",
    "BasisAwareSpectralProjector",
    "projected_common_basis_stress_divergence",
    "projected_distortion_stress_divergence",
    "FreeSlipModalStokesSolver",
    "BoundedAxisPlanKey",
    "DenseBoundedAxisExecutionPlan",
    "build_dense_orthonormal_matrix",
)


EXPECTED_SIGNATURES = {
    "TransformMetadata": (
        "(boundary_conditions: tuple[str, ...], "
        "transform_kinds: tuple[str, ...], "
        "axis_modes: tuple[torch.Tensor, ...], q2: torch.Tensor, "
        "q2_safe: torch.Tensor, laplacian_eigs: torch.Tensor) -> None"
    ),
    "TensorProductTransformBackend": (
        "(shape, lengths, device='cuda', dtype=torch.float32, "
        "execution_order='real_first', spectral_storage='full_complex', "
        "hermitian_axis=None, periodic_transform_execution='multidim')"
    ),
    "BasisAwareSpectralProjector": (
        "(solver, rule='cubic_half', transform_execution=None)"
    ),
    "projected_common_basis_stress_divergence": (
        "(backend, stress_components, boundary_conditions, *, "
        "projector=None, sum_space='physical')"
    ),
    "projected_distortion_stress_divergence": (
        "(backend, stress_components, even_boundary_conditions, "
        "odd_boundary_conditions, *, projector=None, "
        "sum_space='physical')"
    ),
    "FreeSlipModalStokesSolver": (
        "(backend, *, tangential_boundary_conditions, "
        "normal_boundary_conditions, pressure_boundary_conditions, "
        "friction=0.0, viscosity=1.0, zero_mode_policy='zero_mean', "
        "pressure_diagnostics=True)"
    ),
    "BoundedAxisPlanKey": (
        "(kind: 'str', physical_size: 'int', retained_count: 'int', "
        "device: 'torch.device', real_dtype: 'torch.dtype', "
        "value_type: 'str') -> None"
    ),
    "DenseBoundedAxisExecutionPlan": (
        "(key: 'BoundedAxisPlanKey', matrix: 'torch.Tensor') -> None"
    ),
    "build_dense_orthonormal_matrix": (
        "(kind: 'str', size: 'int', *, device: 'torch.device | str', "
        "dtype: 'torch.dtype') -> 'torch.Tensor'"
    ),
}


EXPECTED_IMPLEMENTATION_MODULES = {
    "TransformMetadata": "pssolver.transforms",
    "TensorProductTransformBackend": "pssolver.transforms",
    "BasisAwareSpectralProjector": "pssolver.transforms",
    "projected_common_basis_stress_divergence": "pssolver.transforms",
    "projected_distortion_stress_divergence": "pssolver.transforms",
    "FreeSlipModalStokesSolver": "pssolver.transforms",
    "BoundedAxisPlanKey": "pssolver.backends.bounded",
    "DenseBoundedAxisExecutionPlan": "pssolver.backends.bounded",
    "build_dense_orthonormal_matrix": "pssolver.backends.bounded",
}


ROOT_REEXPORTS = (
    "BasisAwareSpectralProjector",
    "DEALIAS_RULE_FRACTIONS",
    "DEFAULT_DEALIAS_RULE",
    "DEFAULT_PROJECTED_TRANSFORM_EXECUTION",
    "DEFAULT_PERIODIC_TRANSFORM_EXECUTION",
    "DEFAULT_SPECTRAL_STORAGE",
    "DEFAULT_TRANSFORM_EXECUTION_ORDER",
    "FreeSlipModalStokesSolver",
    "TensorProductTransformBackend",
    "PROJECTED_TRANSFORM_EXECUTION_MODES",
    "PERIODIC_TRANSFORM_EXECUTION_MODES",
    "SPECTRAL_STORAGE_MODES",
    "projected_common_basis_stress_divergence",
    "projected_distortion_stress_divergence",
)


EXPECTED_METHOD_SIGNATURES = {
    "TensorProductTransformBackend.periodic_transform_execution_metadata": (
        "(self, boundary_conditions)"
    ),
    "TensorProductTransformBackend.get_metadata": (
        "(self, boundary_conditions)"
    ),
    "TensorProductTransformBackend.get_q2": (
        "(self, boundary_conditions, regularize=False)"
    ),
    "TensorProductTransformBackend.get_laplacian_eigs": (
        "(self, boundary_conditions)"
    ),
    "TensorProductTransformBackend.get_gradient_boundary_conditions": (
        "(self, boundary_conditions, axis)"
    ),
    "TensorProductTransformBackend.forward": (
        "(self, tensor, boundary_conditions, *, retained_axis_counts=None)"
    ),
    "TensorProductTransformBackend.inverse": (
        "(self, spectral, boundary_conditions, *, retained_axis_counts=None)"
    ),
    "TensorProductTransformBackend.laplacian_hat": (
        "(self, spectral, boundary_conditions)"
    ),
    "TensorProductTransformBackend.gradient_hat": (
        "(self, spectral, boundary_conditions, axis)"
    ),
    "BasisAwareSpectralProjector.mask": "(self, boundary_conditions)",
    "BasisAwareSpectralProjector.project": (
        "(self, spectral, boundary_conditions)"
    ),
    "BasisAwareSpectralProjector.project_": (
        "(self, spectral, boundary_conditions)"
    ),
    "BasisAwareSpectralProjector.forward_transform": (
        "(self, tensor, boundary_conditions)"
    ),
    "BasisAwareSpectralProjector.inverse_transform": (
        "(self, spectral, boundary_conditions)"
    ),
    "BasisAwareSpectralProjector.execution_metadata": "(self)",
    "BasisAwareSpectralProjector.computed_axis_sizes": (
        "(self, boundary_conditions)"
    ),
    "BasisAwareSpectralProjector.project_dynamic_fields": (
        "(self, fields, *, sync_spatial)"
    ),
    "BasisAwareSpectralProjector.refresh_dynamic_fields": (
        "(self, fields, *, sync_spatial)"
    ),
    "BasisAwareSpectralProjector.retained_axis_counts": (
        "(self, boundary_conditions)"
    ),
    "FreeSlipModalStokesSolver.divergence_hat": (
        "(self, ux_hat, uy_hat, uz_hat)"
    ),
    "FreeSlipModalStokesSolver.pressure_gradient_hats": (
        "(self, pressure_hat)"
    ),
    "FreeSlipModalStokesSolver.solve_force_hats": (
        "(self, fx_hat, fy_hat, fz_hat)"
    ),
}


def test_transform_facade_exports_the_frozen_v0_1_2_surface():
    for name in EXPECTED_FACADE_NAMES:
        assert hasattr(transforms, name), name
    if hasattr(transforms, "__all__"):
        assert tuple(transforms.__all__) == EXPECTED_FACADE_NAMES


def test_transform_constants_retain_exact_values_and_root_identity():
    assert transforms.DEFAULT_DEALIAS_RULE == "cubic_half"
    assert transforms.DEFAULT_TRANSFORM_EXECUTION_ORDER == "real_first"
    assert transforms.DEFAULT_PROJECTED_TRANSFORM_EXECUTION == "truncated"
    assert transforms.PROJECTED_TRANSFORM_EXECUTION_MODES == (
        "full",
        "truncated",
    )
    assert transforms.DEFAULT_SPECTRAL_STORAGE == "full_complex"
    assert transforms.SPECTRAL_STORAGE_MODES == (
        "full_complex",
        "hermitian_half",
    )
    assert transforms.DEFAULT_PERIODIC_TRANSFORM_EXECUTION == "multidim"
    assert transforms.PERIODIC_TRANSFORM_EXECUTION_MODES == (
        "axiswise",
        "multidim",
    )
    assert transforms.DEALIAS_RULE_FRACTIONS == {
        "none": None,
        "two_thirds": 2.0 / 3.0,
        "cubic_half": 0.5,
    }
    for name in ROOT_REEXPORTS:
        assert name in pssolver.__all__
        assert getattr(pssolver, name) is getattr(transforms, name)


def test_transform_facade_signatures_remain_frozen():
    observed = {
        name: str(inspect.signature(getattr(transforms, name)))
        for name in EXPECTED_SIGNATURES
    }
    assert observed == EXPECTED_SIGNATURES


def test_transform_public_method_signatures_remain_frozen():
    classes = {
        "TensorProductTransformBackend": transforms.TensorProductTransformBackend,
        "BasisAwareSpectralProjector": transforms.BasisAwareSpectralProjector,
        "FreeSlipModalStokesSolver": transforms.FreeSlipModalStokesSolver,
    }
    observed = {}
    for qualified_name in EXPECTED_METHOD_SIGNATURES:
        class_name, method_name = qualified_name.split(".", 1)
        observed[qualified_name] = str(
            inspect.signature(getattr(classes[class_name], method_name))
        )
    assert observed == EXPECTED_METHOD_SIGNATURES


def test_transform_facade_and_root_reexports_share_object_identity():
    for name in ROOT_REEXPORTS:
        assert getattr(pssolver, name) is getattr(transforms, name)


def test_transform_implementation_provenance_is_explicit():
    observed = {
        name: getattr(transforms, name).__module__
        for name in EXPECTED_IMPLEMENTATION_MODULES
    }
    assert observed == EXPECTED_IMPLEMENTATION_MODULES


def test_legacy_transform_pickle_global_paths_resolve_through_the_facade():
    moved_names = (
        "TransformMetadata",
        "TensorProductTransformBackend",
        "BasisAwareSpectralProjector",
        "projected_common_basis_stress_divergence",
        "projected_distortion_stress_divergence",
        "FreeSlipModalStokesSolver",
    )
    for name in moved_names:
        payload = f"cpssolver.transforms\n{name}\n.".encode("ascii")
        assert pickle.loads(payload) is getattr(transforms, name)


def test_free_slip_stokes_state_schema_is_frozen():
    backend = transforms.TensorProductTransformBackend(
        shape=(4, 4, 4),
        lengths=(1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float64,
    )
    solver = transforms.FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=("periodic", "periodic", "neumann"),
        normal_boundary_conditions=("periodic", "periodic", "dirichlet"),
        pressure_boundary_conditions=("periodic", "periodic", "neumann"),
    )
    expected = (
        "ikx",
        "iky",
        "a_tangential_inv",
        "a_normal_inv",
        "tangential_null_mask",
        "dz_neumann_to_dirichlet",
        "dz_dirichlet_to_neumann",
        "schur_diag_safe",
        "pressure_null_mask",
    )
    assert tuple(dict(solver.named_buffers())) == expected
    assert tuple(solver.state_dict()) == expected


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        parts = [function.attr]
        value = function.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def test_supported_checkpoint_code_does_not_pickle_concrete_classes():
    checkpoint_sources = (
        PROJECT_ROOT / "pssolver" / "snapshots.py",
        PROJECT_ROOT / "pssolver" / "workflows" / "plane_checkpoint.py",
    )
    for path in checkpoint_sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        assert imports.isdisjoint({"pickle", "dill", "cloudpickle"})

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            assert name not in {"torch.save", "torch.load"}
            if name not in {"np.load", "numpy.load", "np.save", "numpy.save"}:
                continue
            allow_pickle = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "allow_pickle"
                ),
                None,
            )
            assert isinstance(allow_pickle, ast.Constant)
            assert allow_pickle.value is False
