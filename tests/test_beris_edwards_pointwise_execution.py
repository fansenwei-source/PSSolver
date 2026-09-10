"""Tests for explicit Beris--Edwards pointwise execution policies."""

from __future__ import annotations

import pytest
import torch

from pssolver.models.active_nematics import (
    BerisEdwardsPointwiseKernels,
    DEFAULT_POINTWISE_EXECUTION,
    beris_edwards_algebraic_stress_components,
    beris_edwards_bulk_molecular_field_components,
    beris_edwards_distortion_stress_components,
    beris_edwards_q_nonlinear_components,
)


def _inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(71)
    shape = (1, 4, 3, 2)

    def random_components(count):
        return tuple(
            torch.randn(shape, generator=generator, dtype=dtype)
            for _ in range(count)
        )

    q = random_components(5)
    h = random_components(5)
    q_gradients = tuple(random_components(5) for _ in range(3))
    velocity = random_components(3)
    velocity_gradients = tuple(random_components(3) for _ in range(3))
    return q, h, q_gradients, velocity, velocity_gradients


def _assert_componentwise_identical(observed, expected):
    assert len(observed) == len(expected)
    for actual, reference in zip(observed, expected):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_eager_is_default_and_matches_public_constitutive_helpers_exactly():
    q, h, q_gradients, velocity, velocity_gradients = _inputs()
    kernels = BerisEdwardsPointwiseKernels()

    assert DEFAULT_POINTWISE_EXECUTION == "eager"
    _assert_componentwise_identical(
        kernels.bulk_molecular_field_components(
            q,
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
        ),
        beris_edwards_bulk_molecular_field_components(
            q,
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
        ),
    )
    _assert_componentwise_identical(
        kernels.algebraic_stress_components(
            q,
            h,
            flow_alignment=0.3,
            active_prefactor=-0.01,
        ),
        beris_edwards_algebraic_stress_components(
            q,
            h,
            flow_alignment=0.3,
            active_prefactor=-0.01,
        ),
    )
    _assert_componentwise_identical(
        kernels.distortion_stress_components(q_gradients, ldg_l1=0.02),
        beris_edwards_distortion_stress_components(
            q_gradients,
            ldg_l1=0.02,
        ),
    )
    _assert_componentwise_identical(
        kernels.q_nonlinear_components(
            q,
            velocity,
            q_gradients,
            velocity_gradients,
            ldg_b_over_gamma=-0.1,
            ldg_c_over_gamma=0.1,
            flow_alignment=0.3,
        ),
        beris_edwards_q_nonlinear_components(
            q,
            velocity,
            q_gradients,
            velocity_gradients,
            ldg_b_over_gamma=-0.1,
            ldg_c_over_gamma=0.1,
            flow_alignment=0.3,
        ),
    )


def test_compile_policy_wraps_exactly_four_fullgraph_static_kernels(monkeypatch):
    calls = []

    def fake_compile(function, **options):
        calls.append((function, options))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    kernels = BerisEdwardsPointwiseKernels("compile")

    assert len(calls) == 4
    assert all(
        options
        == {
            "backend": "inductor",
            "mode": "default",
            "dynamic": False,
            "fullgraph": True,
        }
        for _, options in calls
    )
    metadata = kernels.metadata()
    assert metadata["requested"] == "compile"
    assert metadata["effective"] == "compile"
    assert metadata["fallback_allowed"] is False
    assert metadata["fallback_reason"] is None
    assert metadata["compile"]["enabled"] is True
    assert metadata["compile"]["backend"] == "inductor"
    assert metadata["compile"]["mode"] == "default"
    assert metadata["compile"]["dynamic"] is False
    assert metadata["compile"]["fullgraph"] is True


def test_eager_metadata_does_not_claim_an_effective_compiler():
    metadata = BerisEdwardsPointwiseKernels("eager").metadata()

    assert metadata["requested"] == "eager"
    assert metadata["effective"] == "eager"
    assert metadata["compile"]["enabled"] is False
    assert metadata["compile"]["backend"] is None
    assert metadata["compile"]["mode"] is None
    assert isinstance(metadata["compile"]["torch_version"], str)


@pytest.mark.parametrize(
    "arguments, error_type, message",
    (
        (("unknown",), ValueError, "execution"),
        (("eager",), ValueError, "compile_backend"),
        (("eager",), ValueError, "compile_mode"),
    ),
)
def test_invalid_pointwise_configuration_is_rejected(
    arguments,
    error_type,
    message,
):
    options = {}
    if message == "compile_backend":
        options["compile_backend"] = ""
    elif message == "compile_mode":
        options["compile_mode"] = ""
    with pytest.raises(error_type, match=message):
        BerisEdwardsPointwiseKernels(*arguments, **options)


def test_compile_policy_fails_loudly_when_torch_compile_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="requires torch.compile"):
        BerisEdwardsPointwiseKernels("compile")
