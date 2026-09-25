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
    "TransformMetadata": "pssolver.backends.tensor_product",
    "TensorProductTransformBackend": "pssolver.backends.tensor_product",
    "BasisAwareSpectralProjector": "pssolver.operators.projection",
    "projected_common_basis_stress_divergence": (
        "pssolver.operators.tensor_divergence"
    ),
    "projected_distortion_stress_divergence": (
        "pssolver.operators.tensor_divergence"
    ),
    "FreeSlipModalStokesSolver": (
        "pssolver.linear_solvers.stokes.plane_free_slip"
    ),
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


EXPECTED_ROOT_ALL = (
    "__version__",
    "SpectralSolver",
    "Fields",
    "Parameters",
    "DEFAULT_TRANSFORM_GROUP_INDEXING",
    "TRANSFORM_GROUP_INDEXING_MODES",
    "PDEModel",
    "SemiImplicitEulerIntegrator",
    "CONVENTION_NAME",
    "IDEAL_LOOP_MODE_AXES",
    "classify_ideal_loop",
    "convention_metadata",
    "ideal_loop_axis_angles",
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
    "prepare_new_run_directory",
    "write_run_metadata",
    "SimulationSnapshot",
    "REPRESENTATIVE_ORDERED_S_DEFINITION",
    "apply_snapshot_to_solver",
    "load_snapshot",
    "representative_ordered_S",
    "require_distinct_output_directory",
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
    assert tuple(transforms.__all__) == EXPECTED_FACADE_NAMES


def test_transform_facade_contains_no_numerical_implementation():
    path = PROJECT_ROOT / "pssolver" / "transforms.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert forbidden == ()


def test_transform_constants_retain_exact_values_and_root_identity():
    # P7.7.7--P7.7.10 append reviewed public contracts without narrowing or
    # reordering the frozen v0.1.2 compatibility surface.
    assert tuple(pssolver.__all__[: len(EXPECTED_ROOT_ALL)]) == EXPECTED_ROOT_ALL
    phase7_public_surface = (
        "Simulation",
        "GeneratedInitialCondition",
        "Output",
        "SnapshotInitialCondition",
        "SpectralNumerics",
        "TimeStepping",
        "TorchSpectralExecution",
        "CompiledSimulation",
        "compile_simulation",
        "run_simulation",
        "SimulationDiagnosticProtocol",
        "SimulationObservationProtocol",
        "SimulationResult",
        "SimulationRunStatus",
    )
    observed = tuple(pssolver.__all__[len(EXPECTED_ROOT_ALL) :])
    assert observed[: len(phase7_public_surface)] == phase7_public_surface
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


def test_backend_facade_names_are_the_canonical_objects():
    from pssolver.backends import tensor_product

    names = (
        "DEFAULT_TRANSFORM_EXECUTION_ORDER",
        "DEFAULT_SPECTRAL_STORAGE",
        "SPECTRAL_STORAGE_MODES",
        "DEFAULT_PERIODIC_TRANSFORM_EXECUTION",
        "PERIODIC_TRANSFORM_EXECUTION_MODES",
        "TransformMetadata",
        "TensorProductTransformBackend",
    )
    for name in names:
        assert getattr(transforms, name) is getattr(tensor_product, name)


def test_projection_facade_names_are_the_canonical_objects():
    from pssolver.operators import projection

    names = (
        "DEFAULT_DEALIAS_RULE",
        "DEFAULT_PROJECTED_TRANSFORM_EXECUTION",
        "PROJECTED_TRANSFORM_EXECUTION_MODES",
        "DEALIAS_RULE_FRACTIONS",
        "BasisAwareSpectralProjector",
    )
    for name in names:
        assert getattr(transforms, name) is getattr(projection, name)


def test_divergence_facade_names_are_the_canonical_objects():
    from pssolver.operators import tensor_divergence

    names = (
        "projected_common_basis_stress_divergence",
        "projected_distortion_stress_divergence",
    )
    for name in names:
        assert getattr(transforms, name) is getattr(tensor_divergence, name)


def test_stokes_facade_name_is_the_canonical_object():
    from pssolver.linear_solvers.stokes.plane_free_slip import (
        FreeSlipModalStokesSolver,
    )

    assert transforms.FreeSlipModalStokesSolver is FreeSlipModalStokesSolver


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
    expected = {
        "ikx": ((1, 4, 4, 1), torch.complex128),
        "iky": ((1, 4, 4, 1), torch.complex128),
        "a_tangential_inv": ((1, 4, 4, 4), torch.float64),
        "a_normal_inv": ((1, 4, 4, 4), torch.float64),
        "tangential_null_mask": ((1, 4, 4, 4), torch.bool),
        "dz_neumann_to_dirichlet": ((4, 4), torch.complex128),
        "dz_dirichlet_to_neumann": ((4, 4), torch.complex128),
        "schur_diag_safe": ((1, 4, 4, 4), torch.float64),
        "pressure_null_mask": ((1, 4, 4, 4), torch.bool),
    }
    buffers = dict(solver.named_buffers())
    assert tuple(buffers) == tuple(expected)
    assert tuple(solver.state_dict()) == tuple(expected)
    assert {
        name: (tuple(value.shape), value.dtype)
        for name, value in buffers.items()
    } == expected


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
        torch_module_aliases = set()
        torch_save_load_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                torch_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
                if node.module == "torch":
                    torch_save_load_aliases.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in {"save", "load"}
                    )
        assert imports.isdisjoint({"pickle", "dill", "cloudpickle"})

        forbidden_torch_calls = torch_save_load_aliases | {
            f"{alias}.{operation}"
            for alias in torch_module_aliases
            for operation in ("save", "load")
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            assert name not in forbidden_torch_calls
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
