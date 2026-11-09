"""Synthetic, fail-closed tests for RC4.4.7 H100 qualification support."""

from __future__ import annotations

import torch

from benchmarks.qualify_rc447_pressure_cuda import (
    Gate,
    _fields_byte_identical,
)
from benchmarks.run_rc447_cuda_tests import NODE_IDS, classify


def _cuda_tests() -> dict[str, object]:
    outcomes = {node: "passed" for node in NODE_IDS}
    return {
        "schema": "pssolver.rc4_4_7.cuda_only_tests.v1",
        "classification": "PASS_RC4_4_7_SEVEN_CUDA_ONLY_TESTS",
        "passed": True,
        "pytest_returncode": 0,
        "outcomes": outcomes,
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_after_policy": {"matmul": False, "cudnn": False},
            "source_shadow_import": False,
        },
    }


def test_gate_persists_a_failed_check_without_claiming_pass(tmp_path):
    gate = Gate(tmp_path, "synthetic")
    gate.check("positive", True, 1, 1)
    gate.check("negative", False, 2, 1)
    loaded = gate.finish()
    assert loaded["passed"] is False
    assert [item["passed"] for item in loaded["checks"]] == [True, False]


def test_restart_field_comparison_uses_public_keys_and_named_lookup():
    class FieldsWithoutIteration:
        def __init__(self, values):
            self._values = dict(values)

        def keys(self):
            return self._values.keys()

        def __getitem__(self, name):
            if not isinstance(name, str):
                raise KeyError("field keys must be str")
            return self._values[name]

    left = FieldsWithoutIteration({"Q": torch.tensor([1.0]), "u": torch.tensor([2.0])})
    same_reordered = FieldsWithoutIteration({"u": torch.tensor([2.0]), "Q": torch.tensor([1.0])})
    changed = FieldsWithoutIteration({"Q": torch.tensor([1.0]), "u": torch.tensor([3.0])})
    missing = FieldsWithoutIteration({"Q": torch.tensor([1.0])})

    assert _fields_byte_identical(left, same_reordered) is True
    assert _fields_byte_identical(left, changed) is False
    assert _fields_byte_identical(left, missing) is False


def test_cuda_test_classifier_requires_seven_unique_passes():
    collected = list(NODE_IDS)
    outcomes = {node: "passed" for node in collected}
    assert classify(0, collected, outcomes) == (True, [])
    outcomes[collected[-1]] = "skipped"
    passed, problems = classify(0, collected, outcomes)
    assert passed is False
    assert any("non-pass outcomes" in problem for problem in problems)
    passed, problems = classify(0, collected[:-1], {node: "passed" for node in collected[:-1]})
    assert passed is False
    assert any("collected 6 tests" in problem for problem in problems)
