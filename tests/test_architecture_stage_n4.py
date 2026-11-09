"""CPU qualification for Stage N.4 execution-policy consolidation."""

from __future__ import annotations

from pathlib import Path

import pytest

import pssolver
from pssolver.core import (
    HomogeneousNeumannBC,
    PeriodicBC,
)
from pssolver.execution import (
    AlgebraicExecutionMode,
    AlgebraicExecutionPolicy,
    resolve_algebraic_execution_policy,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


PROJECT_ROOT = Path(__file__).parents[1]


def test_execution_policy_encodes_only_the_valid_monotone_modes():
    expected = (
        (AlgebraicExecutionPolicy.eager(), (False, False, False)),
        (AlgebraicExecutionPolicy.reuse(), (True, False, False)),
        (AlgebraicExecutionPolicy.lazy(), (True, True, False)),
        (AlgebraicExecutionPolicy.batched(), (True, True, True)),
    )
    for policy, flags in expected:
        assert (
            policy.representation_reuse,
            policy.lazy_physical_materialization,
            policy.batched_physical_islands,
        ) == flags
        assert policy.to_metadata()["mode"] == policy.mode.value
    assert AlgebraicExecutionPolicy.batched().mode is (
        AlgebraicExecutionMode.BATCHED_PHYSICAL_ISLANDS
    )
    with pytest.raises(ValueError, match="requires representation reuse"):
        AlgebraicExecutionPolicy.from_legacy_flags(lazy_physical_materialization=True)
    with pytest.raises(ValueError, match="require representation reuse and lazy"):
        AlgebraicExecutionPolicy.from_legacy_flags(
            representation_reuse=True,
            batched_physical_islands=True,
        )


def test_policy_and_legacy_flags_have_one_configuration_authority():
    policy = AlgebraicExecutionPolicy.batched()
    assert resolve_algebraic_execution_policy(policy) is policy
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_algebraic_execution_policy(
            policy,
            enable_algebraic_representation_reuse=False,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        resolve_algebraic_execution_policy(
            None,
            enable_algebraic_representation_reuse=1,
        )


def test_stage_n4_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicExecutionPolicy")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "AlgebraicExecutionPolicy" not in text
        assert "BoundarySignatureTransformScheduler" not in text
