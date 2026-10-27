"""S-10 tests for the shared tensor-free validation primitive."""

from __future__ import annotations

import pytest

from pssolver.core._validation import require_exact_keys


class Rejected(ValueError):
    pass


def _reject(message: str) -> None:
    raise Rejected(message)


def test_exact_keys_returns_an_independent_plain_dict():
    source = {"alpha": 1, "beta": 2}
    result = require_exact_keys(
        source,
        frozenset({"alpha", "beta"}),
        "options",
        reject=_reject,
    )
    assert result == source
    assert result is not source


@pytest.mark.parametrize(
    ("source", "expected_message"),
    (
        ({"alpha": 1}, "missing=('beta',), extra=()"),
        (
            {"alpha": 1, "beta": 2, "gamma": 3},
            "missing=(), extra=('gamma',)",
        ),
    ),
)
def test_exact_keys_delegates_rejection_policy(source, expected_message):
    with pytest.raises(Rejected) as caught:
        require_exact_keys(
            source,
            frozenset({"alpha", "beta"}),
            "options",
            reject=_reject,
        )
    assert expected_message in str(caught.value)
