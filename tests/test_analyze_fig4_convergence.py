import copy
import json

import numpy as np
import pytest

from pssolver.models.active_nematics import Q_convention_metadata
from scripts_plane.analyze_fig4_convergence import (
    RunArtifact,
    analyze_dealias,
    compact_q_frobenius_squared,
    demeaned_relative_l2,
    load_run,
    observed_order,
    q_relative_errors,
    q_scalar_metrics,
    select_geometric_dt_triplet,
    validate_comparability,
    validate_run_identity,
    velocity_scalar_metrics,
)


def test_compact_q_frobenius_squared_counts_qzz_and_off_diagonals():
    # Qzz=-(Qxx+Qyy)=-3, so Q:Q=1+4+9+2*(9+16+25)=114.
    q = np.asarray([1.0, 3.0, 4.0, 2.0, 5.0])
    assert compact_q_frobenius_squared(q) == pytest.approx(114.0)


def test_q_relative_errors_use_full_tensor_frobenius_norm(tmp_path):
    reference = np.zeros((2, 1, 1, 5), dtype=np.float32)
    reference[0, 0, 0] = (1.0, 3.0, 4.0, 2.0, 5.0)
    reference[1, 0, 0] = (0.5, 0.0, 0.0, -0.25, 0.0)
    candidate = 2.0 * reference
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)

    relative_l2, relative_linf = q_relative_errors(
        candidate_path,
        reference_path,
        shape=(2, 1, 1),
        chunk_size=1,
    )

    assert relative_l2 == pytest.approx(1.0)
    assert relative_linf == pytest.approx(1.0)


def test_pressure_relative_error_is_invariant_to_constant_gauge():
    reference = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert demeaned_relative_l2(reference + 123.0, reference) == pytest.approx(0.0)


def test_velocity_metrics_report_vector_and_component_rms(tmp_path):
    velocity = np.zeros((2, 1, 1, 3), dtype=np.float32)
    velocity[..., 0] = 3.0
    velocity[..., 1] = 4.0
    path = tmp_path / "u.npy"
    np.save(path, velocity)

    metrics = velocity_scalar_metrics(path, (2, 1, 1), chunk_size=1)

    assert metrics["u_rms"] == pytest.approx(5.0)
    assert metrics["ux_rms"] == pytest.approx(3.0)
    assert metrics["uy_rms"] == pytest.approx(4.0)
    assert metrics["uz_rms"] == pytest.approx(0.0)
    assert metrics["ux_mean"] == pytest.approx(3.0)
    assert metrics["uy_mean"] == pytest.approx(4.0)


def test_q_scalar_metrics_preserve_float64_order_parameter_precision(tmp_path):
    expected = np.asarray((1.0 / 3.0, 1.0 / 3.0 + 2.0e-10))
    q = np.zeros((2, 1, 1, 5), dtype=np.float64)
    q[..., 0] = expected.reshape(2, 1, 1)
    q[..., 3] = -0.5 * expected.reshape(2, 1, 1)
    path = tmp_path / "Q.npy"
    np.save(path, q)

    metrics = q_scalar_metrics(path, (2, 1, 1), chunk_size=1)

    assert metrics["S_mean"] == pytest.approx(np.mean(expected), abs=1.0e-14)


def test_observed_order_for_halved_first_order_error():
    assert observed_order(0.2, 0.1, 2.0) == pytest.approx(1.0)
    assert observed_order(0.0, 0.0, 2.0) is None


def test_geometric_order_triplet_is_selected_from_four_dt_levels():
    selected = select_geometric_dt_triplet((0.01, 0.005, 0.0025, 0.001))

    assert selected == pytest.approx((0.01, 0.005, 0.0025))
    assert select_geometric_dt_triplet((0.01, 0.006, 0.002)) is None


def test_automatic_order_triplet_never_skips_irregular_dt_levels():
    # A combinatorial search would incorrectly find (0.1, 0.01, 0.001).
    levels = (0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001, 0.0007, 0.0003)

    assert select_geometric_dt_triplet(levels) is None


def test_explicit_order_triplet_may_select_nonadjacent_supplied_levels():
    levels = (0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001)

    assert select_geometric_dt_triplet(
        levels, requested=(0.1, 0.01, 0.001)
    ) == pytest.approx((0.1, 0.01, 0.001))


def test_explicit_order_triplet_must_be_available_ordered_and_geometric():
    levels = (0.01, 0.005, 0.0025, 0.001)

    with pytest.raises(ValueError, match="absent"):
        select_geometric_dt_triplet(levels, requested=(0.02, 0.01, 0.005))
    with pytest.raises(ValueError, match="COARSE > MEDIUM > FINE"):
        select_geometric_dt_triplet(levels, requested=(0.0025, 0.005, 0.01))
    with pytest.raises(ValueError, match="geometrically refined"):
        select_geometric_dt_triplet(levels, requested=(0.01, 0.005, 0.001))


_VARIANT_UNSET = object()


def _write_completed_run(
    tmp_path,
    *,
    script="Plane_beris_edwards_stokes.py",
    variant="beris_edwards_complete_nematic_stress_stokes",
):
    run_dir = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
    run_dir.mkdir()
    shape = (2, 2, 2)
    steps = 1
    model = {
        "name": "active_nematics",
        "Q_convention": Q_convention_metadata(),
        "parameters": {"S_initial": 1.0 / 3.0, "S_bulk": 1.0 / 3.0},
    }
    if variant is not _VARIANT_UNSET:
        model["variant"] = variant
    metadata = {
        "schema_version": 1,
        "script": script,
        "status": "complete",
        "completed_steps": steps,
        "solver": {
            "shape": list(shape),
            "lengths": [10.0, 10.0, 2.0],
            "dt": 0.1,
            "steps": steps,
            "save_interval": steps,
        },
        "model": model,
    }
    if script == "Plane_beris_edwards_stokes.py":
        metadata["validation_config_sha256"] = "f" * 64
        metadata["implementation_provenance"] = {
            "schema_version": 1,
            "files": {
                "Plane_beris_edwards_stokes.py": "0" * 64,
            },
        }

    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    np.save(run_dir / "Q_1.npy", np.zeros((*shape, 5), dtype=np.float64))
    np.save(run_dir / "u_1.npy", np.zeros((*shape, 3), dtype=np.float64))
    np.save(run_dir / "p_1.npy", np.zeros(shape, dtype=np.float64))
    np.save(run_dir / "diagnostics.npy", np.zeros(1, dtype=[("step", np.int64)]))
    return run_dir, metadata


def test_load_run_accepts_complete_beris_edwards_identity(tmp_path):
    run_dir, metadata = _write_completed_run(tmp_path)

    run = load_run(run_dir)
    identity = validate_run_identity(metadata, path=run_dir / "metadata.json")

    assert run.directory == run_dir.resolve()
    assert identity["script"] == "Plane_beris_edwards_stokes.py"
    assert identity["model_variant"] == (
        "beris_edwards_complete_nematic_stress_stokes"
    )


@pytest.mark.parametrize("variant", (_VARIANT_UNSET, "wrong_complete_stress_variant"))
def test_load_run_rejects_missing_or_wrong_beris_edwards_variant(
    tmp_path, variant
):
    run_dir, _ = _write_completed_run(tmp_path, variant=variant)

    with pytest.raises(ValueError, match="model.variant"):
        load_run(run_dir)


@pytest.mark.parametrize(
    "provenance, message",
    (
        (None, "non-empty implementation_provenance"),
        ({}, "non-empty implementation_provenance"),
        ({"schema_version": 1, "files": {}}, "files must be"),
        (
            {"schema_version": 1, "files": {"source.py": "A" * 64}},
            "canonical 64-character",
        ),
        (
            {"schema_version": 1, "files": {"source.py": "0" * 63}},
            "canonical 64-character",
        ),
    ),
)
def test_run_identity_rejects_missing_empty_or_noncanonical_provenance(
    tmp_path, provenance, message
):
    run_dir, metadata = _write_completed_run(tmp_path)

    if provenance is None:
        metadata.pop("implementation_provenance")
    else:
        metadata["implementation_provenance"] = provenance

    with pytest.raises(ValueError, match=message):
        validate_run_identity(metadata, path=run_dir / "metadata.json")


@pytest.mark.parametrize(
    "digest",
    (None, "0" * 63, "A" * 64, "g" * 64),
)
def test_beris_run_identity_requires_canonical_validation_config_hash(
    tmp_path, digest
):
    run_dir, metadata = _write_completed_run(tmp_path)
    if digest is None:
        metadata.pop("validation_config_sha256")
    else:
        metadata["validation_config_sha256"] = digest

    with pytest.raises(ValueError, match="validation_config_sha256"):
        validate_run_identity(metadata, path=run_dir / "metadata.json")


def test_load_run_retains_legacy_benchmark_compatibility(tmp_path):
    run_dir, _ = _write_completed_run(
        tmp_path,
        script="Plane_fig4_benchmark.py",
        variant=_VARIANT_UNSET,
    )

    assert load_run(run_dir).metadata["script"] == "Plane_fig4_benchmark.py"


def test_run_identity_rejects_intermediate_shendruk_script(tmp_path):
    run_dir, metadata = _write_completed_run(
        tmp_path,
        script="Plane_shendruk_stokes.py",
        variant=_VARIANT_UNSET,
    )

    with pytest.raises(ValueError, match="unsupported benchmark script"):
        validate_run_identity(metadata, path=run_dir / "metadata.json")



def _spectral_refresh_metadata(
    *,
    dt,
    steps,
    mode="physical_time",
    requested_time=0.2,
    requested_steps=None,
    effective_steps=20,
    effective_time=0.2,
):
    return {
        "schema_version": 1,
        "script": "Plane_fig4_benchmark.py",
        "status": "complete",
        "completed_steps": steps,
        "solver": {
            "shape": [8, 8, 4],
            "lengths": [10.0, 10.0, 4.0],
            "dt": dt,
            "steps": steps,
            "save_interval": steps,
        },
        "model": {"activity": 0.018},
        "numerics": {
            "dealias": "cubic_half",
            "spectral_refresh": {
                "mode": mode,
                "requested_interval_time": requested_time,
                "requested_interval_steps": requested_steps,
                "effective_interval_steps": effective_steps,
                "effective_interval_time": effective_time,
                "phase_origin_step": 0,
            },
        },
    }


def _artifact(tmp_path, label, metadata):
    solver = metadata["solver"]
    shape = tuple(solver["shape"])
    lengths = tuple(solver["lengths"])
    directory = tmp_path / label
    return RunArtifact(
        label=label,
        directory=directory,
        metadata=metadata,
        shape=shape,
        lengths=lengths,
        dt=solver["dt"],
        steps=solver["steps"],
        final_time=solver["dt"] * solver["steps"],
        q_path=directory / "Q.npy",
        u_path=directory / "u.npy",
        p_path=directory / "p.npy",
        diagnostics_path=directory / "diagnostics.npy",
    )


def test_time_comparison_accepts_physical_refresh_with_different_step_intervals(
    tmp_path,
):
    coarse = _artifact(
        tmp_path,
        "coarse",
        _spectral_refresh_metadata(dt=0.01, steps=100, effective_steps=20),
    )
    fine = _artifact(
        tmp_path,
        "fine",
        _spectral_refresh_metadata(dt=0.005, steps=200, effective_steps=40),
    )

    validate_comparability((coarse, fine), "time")


@pytest.mark.parametrize(
    ("field", "different_value"),
    (
        ("requested_interval_time", 0.1),
        ("effective_interval_time", 0.1),
        ("phase_origin_step", 1),
    ),
)
def test_time_comparison_rejects_other_physical_refresh_differences(
    tmp_path, field, different_value
):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01, steps=100, effective_steps=20
    )
    fine_metadata = _spectral_refresh_metadata(
        dt=0.005, steps=200, effective_steps=40
    )
    fine_metadata["numerics"]["spectral_refresh"][field] = different_value

    with pytest.raises(ValueError):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_time_comparison_keeps_step_refresh_interval_strict(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01,
        steps=100,
        mode="steps",
        requested_time=None,
        requested_steps=20,
        effective_steps=20,
    )
    fine_metadata = _spectral_refresh_metadata(
        dt=0.005,
        steps=200,
        mode="steps",
        requested_time=None,
        requested_steps=40,
        effective_steps=40,
    )

    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_time_comparison_keeps_unrelated_provenance_strict(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01, steps=100, effective_steps=20
    )
    fine_metadata = copy.deepcopy(
        _spectral_refresh_metadata(dt=0.005, steps=200, effective_steps=40)
    )
    fine_metadata["model"]["activity"] = 0.019

    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_space_comparison_ignores_only_grid_dependent_initial_hashes(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(dt=0.005, steps=200)
    coarse_metadata["initial_condition"] = {
        "name": "analytic",
        "seed": 24,
        "raw_q_sha256": "coarse-raw",
        "projected_q_sha256": "coarse-projected",
    }
    fine_metadata = copy.deepcopy(coarse_metadata)
    fine_metadata["solver"]["shape"] = [16, 16, 8]
    fine_metadata["initial_condition"]["raw_q_sha256"] = "fine-raw"
    fine_metadata["initial_condition"]["projected_q_sha256"] = "fine-projected"

    validate_comparability(
        (
            _artifact(tmp_path, "coarse", coarse_metadata),
            _artifact(tmp_path, "fine", fine_metadata),
        ),
        "space",
    )

    fine_metadata["initial_condition"]["seed"] = 25
    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "space",
        )


_DEALIAS_FRACTIONS = {
    "none": None,
    "two_thirds": 2.0 / 3.0,
    "cubic_half": 0.5,
}


def _dealias_metadata(
    rule,
    *,
    projected_hash_character,
    shape=(8, 8, 4),
    dt=0.005,
    steps=200,
):
    metadata = _spectral_refresh_metadata(dt=dt, steps=steps)
    metadata["solver"]["shape"] = list(shape)
    metadata["dealias_rule"] = rule
    metadata["dealias_fraction"] = _DEALIAS_FRACTIONS[rule]
    metadata["numerics"].pop("dealias")
    metadata["numerics"]["dealiasing"] = {
        "rule": rule,
        "fraction": _DEALIAS_FRACTIONS[rule],
        "force_evaluation": "same_for_every_rule",
    }
    metadata["initial_condition"] = {
        "name": "analytic",
        "seed": 24,
        "raw_q_sha256": "a" * 64,
        "projected_q_sha256": projected_hash_character * 64,
    }
    retained = {
        "none": shape,
        "two_thirds": tuple(max(1, int(2 * count / 3)) for count in shape),
        "cubic_half": tuple(max(1, count // 2) for count in shape),
    }[rule]
    metadata["retained_q_modes"] = list(retained)
    metadata["retained_normal_velocity_modes"] = list(retained)
    return metadata


def _dealias_artifacts(tmp_path, *, shape=(8, 8, 4)):
    cubic = _artifact(
        tmp_path,
        "cubic",
        _dealias_metadata(
            "cubic_half",
            projected_hash_character="b",
            shape=shape,
        ),
    )
    two_thirds = _artifact(
        tmp_path,
        "two_thirds",
        _dealias_metadata(
            "two_thirds",
            projected_hash_character="c",
            shape=shape,
        ),
    )
    return cubic, two_thirds


def test_dealias_comparison_allows_only_projection_dependent_provenance(
    tmp_path,
):
    cubic, two_thirds = _dealias_artifacts(tmp_path)

    validate_comparability((cubic, two_thirds), "dealias")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model.activity", 0.019),
        ("initial_condition.raw_q_sha256", "d" * 64),
        (
            "numerics.dealiasing.force_evaluation",
            "different_force_evaluation",
        ),
    ),
)
def test_dealias_comparison_rejects_non_dealias_metadata_changes(
    tmp_path, field, value
):
    cubic_metadata = _dealias_metadata(
        "cubic_half", projected_hash_character="b"
    )
    candidate_metadata = _dealias_metadata(
        "two_thirds", projected_hash_character="c"
    )
    container = candidate_metadata
    parts = field.split(".")
    for name in parts[:-1]:
        container = container[name]
    container[parts[-1]] = value

    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "cubic", cubic_metadata),
                _artifact(tmp_path, "candidate", candidate_metadata),
            ),
            "dealias",
        )


def test_dealias_comparison_requires_same_grid_dt_and_final_time(tmp_path):
    cubic_metadata = _dealias_metadata(
        "cubic_half", projected_hash_character="b"
    )
    different_grid = _dealias_metadata(
        "two_thirds",
        projected_hash_character="c",
        shape=(16, 8, 4),
    )
    with pytest.raises(ValueError, match="common grid shape"):
        validate_comparability(
            (
                _artifact(tmp_path, "cubic", cubic_metadata),
                _artifact(tmp_path, "different_grid", different_grid),
            ),
            "dealias",
        )

    different_dt = _dealias_metadata(
        "two_thirds",
        projected_hash_character="c",
        dt=0.01,
        steps=100,
    )
    with pytest.raises(ValueError, match="common dt"):
        validate_comparability(
            (
                _artifact(tmp_path, "cubic", cubic_metadata),
                _artifact(tmp_path, "different_dt", different_dt),
            ),
            "dealias",
        )

    different_time = _dealias_metadata(
        "two_thirds",
        projected_hash_character="c",
        steps=199,
    )
    with pytest.raises(ValueError, match="one physical time"):
        validate_comparability(
            (
                _artifact(tmp_path, "cubic", cubic_metadata),
                _artifact(tmp_path, "different_time", different_time),
            ),
            "dealias",
        )


def test_dealias_comparison_rejects_inconsistent_duplicate_rule_metadata(
    tmp_path,
):
    cubic_metadata = _dealias_metadata(
        "cubic_half", projected_hash_character="b"
    )
    candidate_metadata = _dealias_metadata(
        "two_thirds", projected_hash_character="c"
    )
    candidate_metadata["numerics"]["dealiasing"]["rule"] = "cubic_half"

    with pytest.raises(ValueError, match="metadata disagree"):
        validate_comparability(
            (
                _artifact(tmp_path, "cubic", cubic_metadata),
                _artifact(tmp_path, "candidate", candidate_metadata),
            ),
            "dealias",
        )


def test_dealias_comparison_requires_distinct_rules_and_cubic_reference(
    tmp_path,
):
    cubic_1 = _artifact(
        tmp_path,
        "cubic_1",
        _dealias_metadata(
            "cubic_half", projected_hash_character="b"
        ),
    )
    cubic_2 = _artifact(
        tmp_path,
        "cubic_2",
        _dealias_metadata(
            "cubic_half", projected_hash_character="c"
        ),
    )
    with pytest.raises(ValueError, match="distinct dealias rules"):
        validate_comparability((cubic_1, cubic_2), "dealias")

    none = _artifact(
        tmp_path,
        "none",
        _dealias_metadata("none", projected_hash_character="d"),
    )
    two_thirds = _artifact(
        tmp_path,
        "two_thirds",
        _dealias_metadata(
            "two_thirds", projected_hash_character="e"
        ),
    )
    with pytest.raises(ValueError, match="cubic_half reference"):
        validate_comparability((none, two_thirds), "dealias")


def test_analyze_dealias_reports_field_sensitivity_against_cubic_half(
    tmp_path,
):
    shape = (2, 2, 2)
    cubic, two_thirds = _dealias_artifacts(tmp_path, shape=shape)
    for run in (cubic, two_thirds):
        run.directory.mkdir()

    q_reference = np.zeros((*shape, 5), dtype=np.float64)
    q_reference[..., 0] = 1.0
    u_reference = np.ones((*shape, 3), dtype=np.float64)
    p_reference = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    np.save(cubic.q_path, q_reference)
    np.save(cubic.u_path, u_reference)
    np.save(cubic.p_path, p_reference)
    np.save(two_thirds.q_path, 2.0 * q_reference)
    np.save(two_thirds.u_path, 2.0 * u_reference)
    np.save(two_thirds.p_path, 2.0 * p_reference + 17.0)

    validate_comparability((two_thirds, cubic), "dealias")
    scalar_rows = [{"label": two_thirds.label}, {"label": cubic.label}]
    pair_rows, method = analyze_dealias(
        (two_thirds, cubic), scalar_rows, chunk_size=1
    )

    assert method["analysis_kind"] == "dealias_sensitivity"
    assert method["reference_run"] == cubic.label
    assert method["reference_rule"] == "cubic_half"
    assert method["automatic_convergence_claim"] is False
    assert "not a convergence estimate" in method["interpretation"]
    assert len(pair_rows) == 1
    assert pair_rows[0]["candidate"] == two_thirds.label
    assert pair_rows[0]["reference"] == cubic.label
    assert pair_rows[0]["candidate_dealias_rule"] == "two_thirds"
    assert pair_rows[0]["q_rel_l2"] == pytest.approx(1.0)
    assert pair_rows[0]["u_rel_l2"] == pytest.approx(1.0)
    assert pair_rows[0]["p_demeaned_rel_l2"] == pytest.approx(1.0)

    rows_by_label = {row["label"]: row for row in scalar_rows}
    assert rows_by_label[cubic.label]["is_cubic_half_reference"] is True
    assert rows_by_label[two_thirds.label]["is_cubic_half_reference"] is False
    assert rows_by_label[two_thirds.label]["q_rel_l2_to_reference"] == pytest.approx(
        1.0
    )
