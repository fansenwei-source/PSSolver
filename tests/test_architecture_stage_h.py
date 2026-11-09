"""Characterization tests for Stage H algebraic DAGs and field lifetimes."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import pssolver
from pssolver.core import BoundarySet, DealiasRule, FieldRole, FieldSpec, HomogeneousDirichletBC, HomogeneousNeumannBC, NumericsConfig, PeriodicBC, Precision, ProjectedTransformExecution, SpectralStorage, TransformExecutionOrder
from pssolver.execution import AlgebraicSystemSpec, build_algebraic_execution_plan
from pssolver.models.canary import (
    DifferentialAlgebraicChainModel,
    ScalarDiffusionModel,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


SCALAR_BOUNDARIES = BoundarySet((P, N))


GRADIENT_BOUNDARIES = BoundarySet((P, D))


def _numerics(*, half_spectrum=False):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=(
            DealiasRule.CUBIC_HALF
            if half_spectrum
            else DealiasRule.NONE
        ),
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
            if half_spectrum
            else ProjectedTransformExecution.FULL
        ),
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF
            if half_spectrum
            else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=0 if half_spectrum else None,
    )


def _model():
    return DifferentialAlgebraicChainModel(
        boundaries=SCALAR_BOUNDARIES,
        gradient_boundaries=GRADIENT_BOUNDARIES,
        gradient_axis=1,
        diffusivity=0.07,
        coupling=-0.3,
        helmholtz_shift=1.1,
        helmholtz_length_sq=0.2,
        initial_amplitude=0.25,
        initial_modes=(1, 2),
    )


@dataclass(frozen=True)
class _OperatorRHSModel(ScalarDiffusionModel):
    def explicit_rhs(self, state, context):
        return {"phi": context.laplacian("phi", state["phi"])}


def test_dependency_plan_rejects_missing_owners_duplicate_owners_and_cycles():
    first = AlgebraicSystemSpec(
        name="first",
        capability="first_capability",
        output_components=("a",),
        dependencies=("phi",),
    )
    missing = AlgebraicSystemSpec(
        name="missing",
        capability="missing_capability",
        output_components=("b",),
        dependencies=("unknown",),
    )
    with pytest.raises(ValueError, match="must be evolved or algebraic"):
        build_algebraic_execution_plan(
            (first, missing),
            initial_components=("phi",),
            output_components=("a", "b"),
        )

    duplicate = AlgebraicSystemSpec(
        name="duplicate",
        capability="duplicate_capability",
        output_components=("a",),
        dependencies=("phi",),
    )
    with pytest.raises(ValueError, match="multiple owners"):
        build_algebraic_execution_plan(
            (first, duplicate),
            initial_components=("phi",),
            output_components=("a",),
        )

    cyclic_a = AlgebraicSystemSpec(
        name="cyclic_a",
        capability="capability_a",
        output_components=("a",),
        dependencies=("b",),
    )
    cyclic_b = AlgebraicSystemSpec(
        name="cyclic_b",
        capability="capability_b",
        output_components=("b",),
        dependencies=("a",),
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        build_algebraic_execution_plan(
            (cyclic_a, cyclic_b),
            initial_components=("phi",),
            output_components=("a", "b"),
        )


@dataclass(frozen=True)
class _TransientOnlyModel:
    def field_specs(self):
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, SCALAR_BOUNDARIES),
            FieldSpec.scalar(
                "gradient_phi",
                FieldRole.TRANSIENT,
                GRADIENT_BOUNDARIES,
            ),
        )

    @property
    def name(self):
        return "transient_only"

    def parameter_metadata(self):
        return {}

    def algebraic_system_specs(self):
        return (
            AlgebraicSystemSpec(
                name="gradient",
                capability="component_gradient",
                output_components=("gradient_phi",),
                dependencies=("phi",),
                parameters={"axis": 1},
            ),
        )

    def initial_values(self, context):
        return {
            "phi": torch.zeros(
                context.physical_shape,
                dtype=context.real_dtype,
                device=context.device,
            )
        }

    def linear_operators(self, context):
        return {"phi": torch.zeros_like(context.laplacian_eigenvalues("phi"))}

    def explicit_rhs(self, state, context):
        del context
        return {"phi": state["gradient_phi"]}


def test_stage_h_remains_isolated_from_production_and_benchmark_paths():
    assert not hasattr(pssolver, "AlgebraicExecutionPlan")
    assert not hasattr(pssolver, "MathematicalOperatorContext")
    assert not hasattr(pssolver, "DifferentialAlgebraicChainModel")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "DifferentialAlgebraicChainModel" not in text
        assert "AlgebraicExecutionPlan" not in text
        assert "FieldRole.TRANSIENT" not in text
