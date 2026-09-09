"""Lifecycle tests for single-step Beris--Edwards Q-gradient reuse."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from pssolver.models.active_nematics import BerisEdwardsQGradientCache


def _fields():
    return SimpleNamespace(
        spatial=torch.zeros(9, 1, 3, 3, 3),
        spectral=torch.zeros(9, 1, 3, 3, 3, dtype=torch.complex128),
    )


def _gradients():
    return tuple(
        tuple(torch.full((1,), 10 * axis + component) for component in range(5))
        for axis in range(3)
    )


def test_published_gradients_are_consumed_exactly_once():
    fields = _fields()
    gradients = _gradients()
    cache = BerisEdwardsQGradientCache()

    cache.stage(fields, gradients)
    assert cache.publish(fields) is True
    cached = cache.take(fields)

    assert cached is not None
    assert all(
        actual is expected
        for actual_axis, expected_axis in zip(cached, gradients, strict=True)
        for actual, expected in zip(actual_axis, expected_axis, strict=True)
    )
    assert cache.take(fields) is None


def test_spatial_mutation_invalidates_published_gradients():
    fields = _fields()
    cache = BerisEdwardsQGradientCache()
    cache.stage(fields, _gradients())
    assert cache.publish(fields) is True

    fields.spatial[0].add_(1.0)

    assert cache.take(fields) is None


def test_spectral_mutation_invalidates_published_gradients():
    fields = _fields()
    cache = BerisEdwardsQGradientCache()
    cache.stage(fields, _gradients())
    assert cache.publish(fields) is True

    fields.spectral[0].add_(1.0)

    assert cache.take(fields) is None


def test_wrong_owner_and_unpublished_stage_are_cache_misses():
    fields = _fields()
    other_fields = _fields()
    cache = BerisEdwardsQGradientCache()
    cache.stage(fields, _gradients())

    assert cache.take(fields) is None

    cache.stage(fields, _gradients())
    assert cache.publish(other_fields) is False
    assert cache.take(fields) is None
