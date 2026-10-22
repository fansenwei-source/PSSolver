"""RC4.4.7 recovery gates for the supported Python 3.10 runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tests.v0_3_s03_successor import resolve_s03_successor

from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelPressureMagnitude,
    channel_pressure_magnitude_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_python310_compatibility_recovery.json"
)
PERFORMANCE_REPAIR = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_repair.json"
)
PERFORMANCE_RECOVERY_V10 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_recovery_v10.json"
)
BOUNDED_OPERATOR_RECOVERY_V11 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_bounded_operator_recovery_v11.json"
)
RC5_R59 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R4 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _magnitude(
    *,
    mantissa: float = 0.5,
    exponent: int,
    value: float | None,
) -> ChannelPressureMagnitude:
    return ChannelPressureMagnitude(
        is_zero=False,
        float64_value=value,
        representable_as_positive_float64=value is not None,
        log2_value=float(math.log2(mantissa) + exponent),
        binary_mantissa=mantissa,
        binary_exponent=exponent,
    )


def _zero_magnitude() -> ChannelPressureMagnitude:
    return ChannelPressureMagnitude(
        is_zero=True,
        float64_value=0.0,
        representable_as_positive_float64=False,
        log2_value=None,
        binary_mantissa=None,
        binary_exponent=None,
    )


def test_magnitude_ratio_supports_python_3_10_without_math_exp2(monkeypatch):
    monkeypatch.delattr(math, "exp2", raising=False)
    one = _magnitude(exponent=1, value=1.0)
    two = _magnitude(exponent=2, value=2.0)

    assert channel_pressure_magnitude_ratio(two, one) == 2.0
    assert channel_pressure_magnitude_ratio(one, two) == 0.5


def test_magnitude_ratio_preserves_overflow_and_underflow_contracts():
    one = _magnitude(exponent=1, value=1.0)
    maximum_finite_power = _magnitude(
        exponent=1024,
        value=math.ldexp(0.5, 1024),
    )
    overflowing = _magnitude(exponent=1025, value=None)
    minimum_subnormal = _magnitude(
        exponent=-1073,
        value=math.ldexp(0.5, -1073),
    )
    underflowing = _magnitude(exponent=-1074, value=None)

    assert channel_pressure_magnitude_ratio(maximum_finite_power, one) == (
        math.ldexp(1.0, 1023)
    )
    with pytest.raises(
        ValueError,
        match="normalized pressure magnitude is not finite",
    ):
        channel_pressure_magnitude_ratio(overflowing, one)
    assert channel_pressure_magnitude_ratio(minimum_subnormal, one) == (
        math.ldexp(1.0, -1074)
    )
    assert channel_pressure_magnitude_ratio(underflowing, one) == 0.0


def test_magnitude_ratio_preserves_zero_contracts():
    zero = _zero_magnitude()
    one = _magnitude(exponent=1, value=1.0)

    assert channel_pressure_magnitude_ratio(zero, zero) == 0.0
    assert channel_pressure_magnitude_ratio(zero, one) == 0.0
    with pytest.raises(ValueError, match="cannot be normalized by zero"):
        channel_pressure_magnitude_ratio(one, zero)


def test_recovery_record_binds_failure_and_exact_successor_sources():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["phase"] == "RC4.4.7-recovery"
    assert record["classification"] == (
        "PASS_RC4_4_7_PYTHON_3_10_COMPATIBILITY_FIX_CPU"
    )
    assert record["failed_h100_job_id"] == "10875720"
    assert record["failed_h100_classification"] == (
        "FAIL_RC4_4_7_PROVIDER_PYTHON_3_10_MATH_EXP2_COMPATIBILITY"
    )
    successors = json.loads(
        PERFORMANCE_REPAIR.read_text(encoding="utf-8")
    )["successor_sources"]
    v10 = json.loads(
        PERFORMANCE_RECOVERY_V10.read_text(encoding="utf-8")
    )["successor_sources"]
    v11 = json.loads(
        BOUNDED_OPERATOR_RECOVERY_V11.read_text(encoding="utf-8")
    )["successor_sources"]
    r5_9 = json.loads(RC5_R59.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r2 = json.loads(RC6_R2.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r4 = json.loads(RC6_R4.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    for relative, identity in record["successor_sources"].items():
        actual = _sha256(ROOT / relative)
        if actual == identity["current_sha256"]:
            continue
        repaired = successors[relative]
        successor = v11.get(relative, v10.get(relative, repaired))
        assert repaired["historical_sha256"] == identity["current_sha256"]
        if relative in v11:
            assert v10[relative]["historical_sha256"] == repaired["current_sha256"]
            assert successor["historical_sha256"] == v10[relative]["current_sha256"]
            assert successor["introduced_by_phase"] == (
                "RC4.4.7-bounded-operator-recovery-v11-provider"
            )
        elif relative in v10:
            assert successor["historical_sha256"] == repaired["current_sha256"]
            assert successor["introduced_by_phase"] == (
                "RC4.4.7-performance-recovery-v10-provider"
            )
        else:
            assert successor["introduced_by_phase"] == (
                "RC4.4.7-performance-repair-v9-provider"
            )
        resolved = successor["current_sha256"]
        latest = r5_9.get(relative)
        if latest is not None and latest["historical_sha256"] == resolved:
            resolved = latest["current_sha256"]
        latest = rc6_r2.get(relative)
        if latest is not None and latest["historical_sha256"] == resolved:
            resolved = latest["current_sha256"]
        latest = rc6_r4.get(relative)
        if latest is not None and latest["historical_sha256"] == resolved:
            resolved = latest["current_sha256"]
        assert resolve_s03_successor(relative, resolved) == actual
    assert record["successor_sources"][
        "pssolver/linear_solvers/stokes/channel_no_slip.py"
    ]["historical_sha256"] == (
        "d77b29b0834a35e8e8d41d31cad75018765272880570ce8cb293f359774aeb75"
    )
    assert record["completion"]["python_3_10_runtime_compatibility_fixed"] is True
    assert record["completion"]["H100_requalified"] is False
    assert record["completion"]["production_default_changed"] is False
