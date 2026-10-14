"""Executable RC4.5.4 finite-Q and Robin support-bound contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pssolver import Simulation, compile_simulation
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.finite_q_anchoring import (
    apply_plane_finite_q_anchoring_pilot,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.configuration.robin import (
    validate_public_robin_impedance_support,
)
from pssolver.core.boundary import BoundarySide, StaticRobinCoefficients
from pssolver.models.active_nematics import (
    finite_homeotropic_q_anchoring,
    finite_planar_q_anchoring,
    quadratic_finite_q_anchoring,
    uniaxial_Q,
)
from pssolver.models.active_nematics.boundaries import _oriented_face_normal
from pssolver.planning.robin import (
    MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE,
    _characteristic,
    _positive_robin_roots,
    build_cell_centered_robin_eigenbasis_plan,
    validate_dimensionless_robin_impedance,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"


def _base_simulation(tmp_path: Path):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "must_not_materialize",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=12,
        ny=10,
        nz=8,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )
    return compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )


def _public_simulation(specification) -> Simulation:
    return Simulation(
        model=specification.equation_system,
        geometry=specification.geometry,
        boundaries=specification.boundaries,
        numerics=specification.numerics,
        time=specification.time_integration,
        initial_condition=specification.initial_condition,
        execution=specification.execution,
        output=specification.workflow,
        discretization=specification.discretization_parameters,
        invocation=specification.invocation,
    )


def _normal_map(sign: float = 1.0):
    return {
        (2, "lower"): (0.0, 0.0, -sign),
        (2, "upper"): (0.0, 0.0, sign),
    }


def test_finite_q_homeotropic_normal_sign_is_one_canonical_identity():
    common = {
        "k_q": 0.02,
        "wall_strengths": {(2, "lower"): 0.1, (2, "upper"): 0.2},
        "scalar_order": 0.6,
    }
    outward = finite_homeotropic_q_anchoring(
        **common,
        face_normals=_normal_map(1.0),
    )
    opposite = finite_homeotropic_q_anchoring(
        **common,
        face_normals=_normal_map(-1.0),
    )

    assert outward.to_metadata() == opposite.to_metadata()
    assert outward.canonical_sha256() == opposite.canonical_sha256()
    assert outward.to_robin_policy() == opposite.to_robin_policy()
    for left, right in zip(outward.faces, opposite.faces):
        assert np.asarray(left.target_components).tobytes() == np.asarray(
            right.target_components
        ).tobytes()


def test_finite_q_planar_normal_sign_is_one_canonical_identity():
    common = {
        "k_q": 0.02,
        "wall_strengths": {(2, "lower"): 0.1, (2, "upper"): 0.2},
        "scalar_order": 0.6,
        "face_directors": {
            (2, "lower"): (1.0, 0.0, 0.0),
            (2, "upper"): (0.0, 1.0, 0.0),
        },
    }
    outward = finite_planar_q_anchoring(
        **common,
        face_normals=_normal_map(1.0),
    )
    opposite = finite_planar_q_anchoring(
        **common,
        face_normals=_normal_map(-1.0),
    )
    assert outward.to_metadata() == opposite.to_metadata()
    assert outward.canonical_sha256() == opposite.canonical_sha256()


@pytest.mark.parametrize(
    "observed",
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, float("nan"), 1.0),
    ),
)
def test_invalid_anchoring_normal_reports_the_frozen_error_details(observed):
    with pytest.raises(ValueError) as caught:
        _oriented_face_normal(
            (2, BoundarySide.LOWER),
            observed,
            "finite-Q face normal",
        )
    message = str(caught.value)
    for fragment in (
        "axis=2",
        "side='lower'",
        "expected_outward_normal=",
        "observed_normal=",
        "accepted_equivalence='parallel_or_antiparallel'",
    ):
        assert fragment in message


@pytest.mark.parametrize(
    "value",
    (-1.0, 1.0e14 + 1.0, float("nan"), float("inf"), True),
)
def test_dimensionless_robin_impedance_rejects_outside_frozen_domain(value):
    with pytest.raises(ValueError, match=r"\[0, 1e14\]"):
        validate_dimensionless_robin_impedance(value, "test impedance")


def test_robin_support_bound_is_closed_and_preserves_existing_root_bytes():
    assert MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE == 1.0e14
    assert validate_dimensionless_robin_impedance(0.0, "lower") == 0.0
    assert validate_dimensionless_robin_impedance(1.0e14, "upper") == 1.0e14
    assert _positive_robin_roots(8, 1.0, 1.0) == (
        1.3065423741888067,
        3.673194406304253,
        6.584620042564175,
        9.631684635691872,
        12.723240784131333,
        15.834105369332411,
        18.954971410841594,
        22.081659635942582,
    )


_REFERENCE_ROOTS_80_DIGITS = {
    (1.0, 1.0): (
        "1.30654237418880620222872783192311828484127975563506919322803272004605439556623294",
        "3.67319440630425144550260500488202063163741963724860798209264923424725269757473619",
        "6.58462004256417319218955036772483655942280637379612346800850213183527219348375090",
        "9.63168463569187088212949838471614686760972539745025469103638128334846400134510382",
    ),
    (1000.0, 1000.0): (
        "3.13532203007683923807425396285517339808173887511625317862710032805338783330468157",
        "6.27064418318790927383581055888625187909461215286602795026695542427300289407658133",
        "9.40596658235297669926209866571108380784940934399347487380483391651522049331328406",
        "12.5412893505628839423554925174224148524324675210609745555016934493001214424300652",
    ),
    (1.0e14, 1.0e14): (
        "3.14159265358973040660957158867137069276541664281237071128195426189168006370068378",
        "6.28318530717946081321914317734274138553095731073146261192260189134781495363489814",
        "9.42477796076919121982871476601411207829674602886399689128063625593139060154920452",
        "12.5663706143589216264382863546854827710629068223166947387147507232024551502162261",
    ),
    (0.0, 1.0e14): (
        "1.57079632679488091126805374283063876156115768522813231136436948049354816401503495",
        "4.71238898038464273380416122849191628468350406196107723267303285058370388974521697",
        "7.85398163397440455634026871415319380780594345752406304972146944798267058495399309",
        "10.9955742875641663788763761998144713309285378844704503596695280908950981505546183",
    ),
}


@pytest.mark.parametrize("impedances", tuple(_REFERENCE_ROOTS_80_DIGITS))
def test_robin_roots_match_frozen_80_digit_reference(impedances):
    lower, upper = impedances
    roots = _positive_robin_roots(4, lower, upper)
    references = tuple(float(value) for value in _REFERENCE_ROOTS_80_DIGITS[impedances])
    assert len(roots) == 4
    assert all(math.isfinite(value) and value > 0.0 for value in roots)
    assert all(left < right for left, right in zip(roots, roots[1:]))
    for index, (root, reference) in enumerate(zip(roots, references)):
        assert index * math.pi < root < (index + 1) * math.pi
        assert abs(_characteristic(root, lower, upper)) <= 1.0e-12
        assert abs(root - reference) <= 5.0e-12


def test_pure_neumann_retains_its_mathematical_zero_mode():
    plan = build_cell_centered_robin_eigenbasis_plan(
        size=8,
        length=2.0,
        lower=StaticRobinCoefficients(0.0, 1.0, 0.0),
        upper=StaticRobinCoefficients(0.0, 1.0, 0.0),
    )
    assert plan.dimensionless_roots[0] == 0.0
    assert plan.dimensionless_roots[1:] == tuple(
        index * math.pi for index in range(1, 8)
    )


def test_public_above_bound_rejects_before_output_or_root_materialization(
    tmp_path,
    monkeypatch,
):
    base = _base_simulation(tmp_path)
    length = base.geometry.domain.lengths[2]
    k_q = base.equation_system.parameters["ldg_l1"]
    wall_strength = (1.0e14 + 1.0) * k_q / length
    target = uniaxial_Q(np.asarray((0.0, 0.0, 1.0)), 0.6)
    anchoring = quadratic_finite_q_anchoring(
        k_q=k_q,
        wall_strengths={(2, "lower"): wall_strength, (2, "upper"): 0.1},
        target_q={(2, "lower"): target, (2, "upper"): target},
    )
    source = apply_plane_finite_q_anchoring_pilot(base, anchoring)

    called = False

    def forbidden_root_materialization(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("root materialization must not start")

    monkeypatch.setattr(
        "pssolver.planning.robin._positive_robin_roots",
        forbidden_root_materialization,
    )
    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(_public_simulation(source))

    assert caught.value.code is PublicCompilationRejectionCode.BOUNDARY_CONTRACT
    context = caught.value.context["robin_impedance"]
    assert context["axis"] == 2
    assert context["side"] == "lower"
    assert context["dimensionless_impedance"] > 1.0e14
    assert context["qualified_upper_inclusive"] == 1.0e14
    assert caught.value.__cause__.__class__.__name__ == (
        "RobinImpedanceSupportError"
    )
    assert called is False
    assert not (tmp_path / "must_not_materialize").exists()


def test_public_bound_validator_accepts_exact_upper_endpoint(tmp_path):
    base = _base_simulation(tmp_path)
    length = base.geometry.domain.lengths[2]
    k_q = base.equation_system.parameters["ldg_l1"]
    wall_strength = 1.0e14 * k_q / length
    target = uniaxial_Q(np.asarray((0.0, 0.0, 1.0)), 0.6)
    anchoring = quadratic_finite_q_anchoring(
        k_q=k_q,
        wall_strengths={(2, "lower"): wall_strength, (2, "upper"): 0.0},
        target_q={(2, "lower"): target, (2, "upper"): target},
    )
    source = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    validate_public_robin_impedance_support(source)


def test_rc454_record_and_matrix_close_only_b9_and_b21():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_RC4_5_4_BOUNDARY_SEMANTICS_CPU_CANDIDATE"
    )
    assert record["issues"] == {"B9": "closed_verified", "B21": "closed_verified"}
    assert record["scope"]["B13_B15_B17_modified"] is False
    assert record["scope"]["B11_modified"] is False
    assert record["scope"]["h100_executed"] is False
    assert record["authorization"]["eligible_for_rc4_5_5_implementation"] is True
    assert record["authorization"]["rc4_5_5_implementation_authorized"] is False
    issues = {value["id"]: value for value in matrix["items"]}
    assert issues["B9"]["state"] == "closed_verified"
    assert issues["B21"]["state"] == "closed_verified"
    assert matrix["progress"]["rc4_5_4_complete"] is True
