from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import scripts_plane.analyze_beris_edwards_stationarity as stationarity
from scripts_plane.analyze_beris_edwards_stationarity import (
    CLASS_INCONCLUSIVE,
    CLASS_NONSTATIONARY,
    CLASS_PROVISIONAL,
    SpectralElasticEnergy,
    _compact_q_statistics,
    _velocity_statistics,
    analyze,
    summarize_series,
)
from scripts_plane.run_beris_edwards_validation import (
    IMPLEMENTATION_SOURCE_FILES,
    OUTPUT_VALIDATOR,
    PROJECT_ROOT,
    RunSpec,
    _canonical_sha256,
    _expected_metadata_fields,
    _sha256_file,
)
from scripts_plane.validate_beris_edwards_outputs import (
    DIAGNOSTIC_FIELDS,
    RUNNER_PATH,
    validate_outputs,
)


DIAGNOSTIC_DTYPE = [
    ("step", np.int64),
    ("div_max", np.float64),
    ("div_rms", np.float64),
    ("div_rel", np.float64),
    ("schur_iterations", np.float64),
    ("schur_abs_residual", np.float64),
    ("schur_rel_residual", np.float64),
    ("wall_normal_momentum_max", np.float64),
    ("wall_normal_momentum_rms", np.float64),
]


def _set_nested(payload, path, value):
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_validated_long_run(
    tmp_path: Path,
    *,
    velocity=(3.0, 4.0, 0.0),
    scalar_order=1.0 / 3.0,
):
    output_root = tmp_path / "simulation"
    run_dir = output_root / "synthetic_long_pilot"
    control_dir = tmp_path / "control"
    run_dir.mkdir(parents=True)
    control_dir.mkdir()
    shape = (2, 3, 4)
    final_step = 20
    spec = RunSpec(
        run_id="synthetic_long_pilot",
        purposes=("long_pilot",),
        activity_number=18.0,
        seed=24,
        nx=shape[0],
        ny=shape[1],
        nz=shape[2],
        dt=0.5,
        steps=final_step,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        dealias_rule="cubic_half",
    )
    config = spec.scientific_config()
    config.update(
        {"dtype": "float64", "device": "cpu", "expected_gpu_name": None}
    )
    row = {
        "run_id": spec.run_id,
        "purposes": ["long_pilot"],
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "output_dir": str(run_dir),
    }
    implementation = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in IMPLEMENTATION_SOURCE_FILES
    }
    metadata = {}
    for path, value in _expected_metadata_fields(row).items():
        _set_nested(metadata, path, value)
    metadata["model"]["parameters"]["S_bulk"] = scalar_order
    metadata["S_bulk"] = scalar_order
    metadata["boundary_conditions"] = {"Q": ["periodic", "periodic", "neumann"]}
    metadata["initial_condition"]["paper_identical"] = False
    metadata["implementation_provenance"] = {
        "schema_version": 1,
        "files": implementation,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, allow_nan=False), encoding="utf-8"
    )
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")

    q = np.zeros((*shape, 5), dtype=np.float64)
    q[..., 0] = scalar_order
    q[..., 3] = -0.5 * scalar_order
    u = np.empty((*shape, 3), dtype=np.float64)
    u[...] = velocity
    for step in range(final_step + 1):
        np.save(run_dir / f"Q_{step}.npy", q)
        np.save(run_dir / f"u_{step}.npy", u)
        np.save(run_dir / f"p_{step}.npy", np.zeros(shape, dtype=np.float64))

    diagnostics = np.zeros(final_step + 1, dtype=DIAGNOSTIC_DTYPE)
    diagnostics["step"] = np.arange(final_step + 1)
    diagnostics["div_max"] = 2.0e-12
    diagnostics["div_rms"] = 1.0e-12
    diagnostics["div_rel"] = 2.0e-13
    diagnostics["schur_iterations"] = 1.0
    diagnostics["schur_abs_residual"] = 3.0e-13
    diagnostics["schur_rel_residual"] = 4.0e-13
    diagnostics["wall_normal_momentum_max"] = 5.0e-12
    diagnostics["wall_normal_momentum_rms"] = 6.0e-13
    np.save(run_dir / "diagnostics.npy", diagnostics)
    with (run_dir / "diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(DIAGNOSTIC_FIELDS)
        writer.writerows(diagnostics.tolist())

    plan = {
        "schema_version": 1,
        "validation": "beris_edwards_stokes_issue6",
        "output_root": str(output_root),
        "implementation_sha256": implementation,
        "validation_tools_sha256": {
            str(OUTPUT_VALIDATOR.relative_to(PROJECT_ROOT)): _sha256_file(
                OUTPUT_VALIDATOR
            ),
            str(RUNNER_PATH.relative_to(PROJECT_ROOT)): _sha256_file(RUNNER_PATH),
        },
        "advisory_gates": {
            "divergence_and_schur_relative_residual_max": 1.0e-10
        },
        "runs": [row],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    plan_path = output_root / "validation_plan.json"
    plan_path.write_text(json.dumps(plan, allow_nan=False), encoding="utf-8")

    validation = validate_outputs(
        run_dir, plan_path=plan_path, run_id=spec.run_id
    )
    assert validation["passed"], validation["errors"]
    validation_path = control_dir / "postvalidation.json"
    validation_path.write_text(
        json.dumps(validation, allow_nan=False), encoding="utf-8"
    )
    checksum_path = control_dir / "checksum_manifest.json"
    manifest_targets = [
        plan_path,
        validation_path,
        run_dir / "COMPLETE",
        run_dir / "metadata.json",
        run_dir / "diagnostics.npy",
        run_dir / "diagnostics.csv",
        *(
            run_dir / f"{field}_{step}.npy"
            for step in range(final_step + 1)
            for field in ("Q", "u", "p")
        ),
    ]
    manifest_entries = []
    for path in manifest_targets:
        stat = path.stat()
        manifest_entries.append(
            {
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    checksum_payload = {
        "schema_version": 1,
        "phase": "synthetic_test",
        "entries": manifest_entries,
    }
    checksum_manifest = {
        **checksum_payload,
        "payload_sha256": _canonical_sha256(checksum_payload),
    }
    checksum_path.write_text(
        json.dumps(checksum_manifest, allow_nan=False),
        encoding="utf-8",
    )
    return run_dir, plan_path, validation_path, checksum_path, spec.run_id


def _analysis_kwargs(tmp_path: Path):
    run_dir, plan, validation, checksum, run_id = _write_validated_long_run(
        tmp_path
    )
    return {
        "run_dir": run_dir,
        "plan_path": plan,
        "run_id": run_id,
        "validation_report_path": validation,
        "checksum_manifest_path": checksum,
        "expected_validation_report_sha256": _sha256_file(validation),
        "expected_checksum_manifest_sha256": _sha256_file(checksum),
        "output_dir": tmp_path / "analysis",
        "include_defects": False,
        "energy_device": "cpu",
        "chunk_x": 1,
        "pass_relative_drift": 0.10,
        "fail_relative_drift": 0.20,
        "min_effective_samples": 5.0,
        "min_finite_frames": 15,
        "max_acf_lag": 7,
        "defect_threshold": 0.0,
        "wall_tolerance_index": 0.51,
        "require_clean_git": False,
        "expected_analysis_git_head": None,
    }


def test_canonical_q_convention_invariants_and_bulk_energy(tmp_path):
    shape = (2, 2, 2)
    scalar_order = 0.4
    q = np.zeros((*shape, 5), dtype=np.float64)
    q[0, ..., 0] = scalar_order
    q[0, ..., 3] = -0.5 * scalar_order
    q[1, ..., 0] = 0.25 * scalar_order
    q[1, ..., 1] = 0.75 * scalar_order
    q[1, ..., 3] = 0.25 * scalar_order
    path = tmp_path / "Q.npy"
    np.save(path, q)
    coefficients = {"ldg_a": 0.2, "ldg_b": -0.3, "ldg_c": 0.4}

    result = _compact_q_statistics(
        path,
        shape=shape,
        dtype=np.dtype("float64"),
        s_bulk=scalar_order,
        chunk_x=1,
        **coefficients,
    )

    expected_bulk = (
        coefficients["ldg_a"] * 3.0 * scalar_order**2 / 4.0
        + coefficients["ldg_b"] * scalar_order**3 / 4.0
        + coefficients["ldg_c"] * 9.0 * scalar_order**4 / 16.0
    )
    assert result["principal_S_mean"] == pytest.approx(scalar_order)
    assert result["Q_magnitude_mean"] == pytest.approx(scalar_order)
    assert result["principal_S_mean_over_S_bulk"] == pytest.approx(1.0)
    assert result["Q_magnitude_mean_over_S_bulk"] == pytest.approx(1.0)
    assert result["biaxiality_mean"] == pytest.approx(0.0, abs=2e-15)
    assert result["bulk_ldg_free_energy_density_mean"] == pytest.approx(
        expected_bulk
    )


def test_velocity_rms_is_sqrt_mean_speed_squared(tmp_path):
    shape = (2, 3, 2)
    velocity = np.zeros((*shape, 3), dtype=np.float64)
    velocity[...] = (3.0, 4.0, 0.0)
    path = tmp_path / "u.npy"
    np.save(path, velocity)

    result = _velocity_statistics(
        path, shape=shape, dtype=np.dtype("float64"), chunk_x=1
    )

    assert result["u_rms"] == pytest.approx(5.0)
    assert result["speed_mean"] == pytest.approx(5.0)
    assert result["velocity_energy_density_mean"] == pytest.approx(12.5)
    assert (
        result["mean_u_x"],
        result["mean_u_y"],
        result["mean_u_z"],
    ) == pytest.approx((3.0, 4.0, 0.0))


def test_constant_q_has_zero_mixed_basis_elastic_energy(tmp_path):
    shape = (6, 5, 4)
    q = np.zeros((*shape, 5), dtype=np.float64)
    q[..., 0] = 0.3
    q[..., 3] = -0.15
    path = tmp_path / "Q.npy"
    np.save(path, q)
    evaluator = SpectralElasticEnergy(
        shape=shape,
        lengths=(2.0, 3.0, 1.5),
        dtype=np.dtype("float64"),
        device="cpu",
        ldg_l1=0.02,
    )

    assert evaluator.evaluate(path) == pytest.approx(0.0, abs=1e-28)


def test_mixed_basis_elastic_energy_matches_periodic_neumann_mode(tmp_path):
    shape = (16, 6, 12)
    lengths = (2.5, 1.7, 1.3)
    amplitude = 0.08
    ldg_l1 = 0.03
    evaluator = SpectralElasticEnergy(
        shape=shape,
        lengths=lengths,
        dtype=np.dtype("float64"),
        device="cpu",
        ldg_l1=ldg_l1,
    )
    x, _, z = evaluator.backend.spatial_grids
    qxx = amplitude * np.cos(
        2.0 * np.pi * x.cpu().numpy() / lengths[0]
    ) * np.cos(np.pi * z.cpu().numpy() / lengths[2])
    q = np.zeros((*shape, 5), dtype=np.float64)
    q[..., 0] = qxx
    q[..., 3] = -0.5 * qxx
    path = tmp_path / "Q_mode.npy"
    np.save(path, q)

    expected = (
        3.0
        * ldg_l1
        * amplitude**2
        * (
            (2.0 * np.pi / lengths[0]) ** 2
            + (np.pi / lengths[2]) ** 2
        )
        / 16.0
    )
    assert evaluator.evaluate(path) == pytest.approx(
        expected, rel=2e-12, abs=2e-14
    )


def test_stationarity_summary_uses_physical_time_and_symmetric_blocks():
    times = np.arange(21, dtype=float) * 2.5 + 50.0
    constant = summarize_series(
        times,
        np.full(21, 3.0),
        pass_relative_drift=0.1,
        fail_relative_drift=0.2,
        min_effective_samples=5,
        min_finite_frames=15,
        max_acf_lag=7,
    )
    assert constant["classification"] == CLASS_PROVISIONAL
    assert constant["slope_per_time"] == pytest.approx(0.0, abs=1e-15)
    assert constant["early_block"] == {
        "count": 10,
        "start_time": 50.0,
        "end_time": 72.5,
        "mean": 3.0,
    }
    assert constant["excluded_midpoint"]["time"] == 75.0
    assert constant["late_block"] == {
        "count": 10,
        "start_time": 77.5,
        "end_time": 100.0,
        "mean": 3.0,
    }

    drifting = summarize_series(
        times,
        1.0 + 0.1 * times,
        pass_relative_drift=0.1,
        fail_relative_drift=0.2,
        min_effective_samples=5,
        min_finite_frames=15,
        max_acf_lag=7,
    )
    assert drifting["classification"] == CLASS_NONSTATIONARY
    assert drifting["slope_per_time"] == pytest.approx(0.1)


def test_missing_fig4_frames_are_inconclusive_not_zero():
    result = summarize_series(
        range(21),
        [None] * 10 + [0.4] * 11,
        pass_relative_drift=0.1,
        fail_relative_drift=0.2,
        min_effective_samples=5,
        min_finite_frames=15,
        max_acf_lag=7,
    )
    assert result["classification"] == CLASS_INCONCLUSIVE
    assert result["finite_frame_count"] == 11
    assert "insufficient_finite_frames" in result["reason_codes"]


def test_full_analysis_is_read_only_and_writes_external_complete_bundle(
    tmp_path,
):
    kwargs = _analysis_kwargs(tmp_path)
    run_dir = kwargs["run_dir"]
    output = kwargs["output_dir"]
    before = _tree_hashes(run_dir)
    control_before = {
        str(path): _sha256_file(path)
        for path in (
            kwargs["plan_path"],
            kwargs["validation_report_path"],
            kwargs["checksum_manifest_path"],
        )
    }

    report = analyze(**kwargs)

    assert _tree_hashes(run_dir) == before
    assert {
        str(path): _sha256_file(path)
        for path in (
            kwargs["plan_path"],
            kwargs["validation_report_path"],
            kwargs["checksum_manifest_path"],
        )
    } == control_before
    manifest_gate = report["input_provenance"][
        "checksum_manifest_verification"
    ]
    assert manifest_gate["status"] == "verified"
    assert manifest_gate["critical_record_count_verified"] == 69
    assert (
        report["classification"]["field_stationarity"]["classification"]
        == CLASS_PROVISIONAL
    )
    assert report["classification"]["overall"] == CLASS_INCONCLUSIVE
    assert report["classification"]["eligible_to_consider_multiseed"] is False
    assert (
        report["analysis_provenance"][
            "analysis_tool_simulation_plan_binding"
        ]
        == "not_applicable_post_hoc"
    )
    assert len(report["frame_observables"]) == 21
    assert (output / "COMPLETE").read_text().strip() == "complete"
    assert {
        "frame_observables.csv",
        "observable_stationarity.csv",
        "defect_lines.csv",
        "stationarity_report.json",
        "stationarity_timeseries.png",
        "analysis_manifest.json",
        "COMPLETE",
    } == {path.name for path in output.iterdir()}
    serialized = (output / "stationarity_report.json").read_text()
    assert "NaN" not in serialized and "Infinity" not in serialized


def test_mocked_defect_metrics_can_complete_all_three_layers(
    tmp_path, monkeypatch
):
    def fake_measure(q, *, step, threshold, wall_tolerance, lengths):
        del q, threshold, wall_tolerance, lengths
        return (
            {
                "defect_point_count": 12,
                "line_count": 3,
                "through_count": 2,
                "bottom_only_count": 0,
                "top_only_count": 0,
                "interior_count": 1,
                "loop_count": 0,
                "mean_sigma_over_h": 0.4,
                "std_sigma_over_h": 0.01,
            },
            [
                {
                    "step": step,
                    "line_index": 0,
                    "sigma": 0.4,
                    "sigma_over_h": 0.4,
                }
            ],
        )

    artifact_path = Path(stationarity.__file__).resolve()
    artifact_sha = _sha256_file(artifact_path)
    source_root = tmp_path / "synthetic_nematics3d"
    source_root.mkdir()
    (source_root / "__init__.py").write_text(
        "synthetic = True\n", encoding="utf-8"
    )
    defect_provenance = {
        "module": "synthetic",
        "implementation_artifacts": [
            {
                "role": "synthetic",
                "path": str(artifact_path),
                "sha256": artifact_sha,
            }
        ],
        "source_tree": stationarity._bind_source_tree(source_root),
        "adapter_path": str(artifact_path),
        "adapter_sha256": artifact_sha,
        "geometry_helper_path": str(artifact_path),
        "geometry_helper_sha256": artifact_sha,
    }
    monkeypatch.setattr(
        stationarity,
        "_load_defect_measurement",
        lambda: (fake_measure, defect_provenance),
    )
    expected_head = "a" * 40
    clean_git = {
        "head": expected_head,
        "expected_head": expected_head,
        "expected_head_matched": True,
        "status_porcelain": "",
        "clean": True,
        "require_clean_requested": True,
        "git_version": "git version synthetic",
        "analysis_tool_simulation_plan_binding": "not_applicable_post_hoc",
    }
    monkeypatch.setattr(
        stationarity,
        "_git_provenance",
        lambda **unused: clean_git,
    )
    kwargs = _analysis_kwargs(tmp_path)
    kwargs.update(
        {
            "include_defects": True,
            "require_clean_git": True,
            "expected_analysis_git_head": expected_head,
        }
    )
    report = analyze(**kwargs)

    assert (
        report["classification"]["field_stationarity"]["classification"]
        == CLASS_PROVISIONAL
    )
    assert (
        report["classification"]["topology_stationarity"]["classification"]
        == CLASS_PROVISIONAL
    )
    assert (
        report["classification"]["fig4_observable_stationarity"][
            "classification"
        ]
        == CLASS_PROVISIONAL
    )
    assert report["classification"]["overall"] == CLASS_PROVISIONAL
    assert report["classification"]["formal_provenance_gate_passed"] is True
    assert report["classification"]["eligible_to_consider_multiseed"] is True
    assert "through_line_count" in report["series"]
    assert "through_line_presence" in report["series"]
    assert all(
        row["through_line_presence"] == 1.0
        for row in report["frame_observables"]
    )
    assert report["thresholds"]["defect_threshold"] == 0.0
    assert report["thresholds"]["wall_tolerance_index"] == 0.51


def test_existing_or_simulation_internal_output_directory_is_rejected(
    tmp_path,
):
    kwargs = _analysis_kwargs(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    kwargs["output_dir"] = existing
    with pytest.raises(FileExistsError, match="wholly new"):
        analyze(**kwargs)

    kwargs["output_dir"] = kwargs["run_dir"].parent / "analysis"
    with pytest.raises(ValueError, match="simulation output_root"):
        analyze(**kwargs)


def test_mismatched_formal_validation_report_is_rejected(tmp_path):
    kwargs = _analysis_kwargs(tmp_path)
    validation = kwargs["validation_report_path"]
    payload = json.loads(validation.read_text())
    payload["runner_plan_binding"] = "mismatch"
    validation.write_text(json.dumps(payload), encoding="utf-8")
    kwargs["expected_validation_report_sha256"] = _sha256_file(validation)

    with pytest.raises(ValueError, match="formal validation report mismatch"):
        analyze(**kwargs)


def test_input_identity_change_during_analysis_is_fatal(
    tmp_path, monkeypatch
):
    kwargs = _analysis_kwargs(tmp_path)
    original = stationarity.SpectralElasticEnergy.evaluate
    calls = 0

    def mutate_once(self, q_path):
        nonlocal calls
        result = original(self, q_path)
        calls += 1
        if calls == 1:
            q_path.touch()
        return result

    monkeypatch.setattr(
        stationarity.SpectralElasticEnergy, "evaluate", mutate_once
    )
    with pytest.raises(RuntimeError, match="critical inputs changed"):
        analyze(**kwargs)
    assert not kwargs["output_dir"].exists()



def test_principal_eigenvalue_matches_random_tensor_eigvalsh(tmp_path):
    rng = np.random.default_rng(20260902)
    shape = (5, 4, 3)
    q = rng.normal(scale=0.2, size=(*shape, 5)).astype(np.float64)
    path = tmp_path / "Q_random.npy"
    np.save(path, q)

    matrices = np.zeros((*shape, 3, 3), dtype=np.float64)
    matrices[..., 0, 0] = q[..., 0]
    matrices[..., 0, 1] = matrices[..., 1, 0] = q[..., 1]
    matrices[..., 0, 2] = matrices[..., 2, 0] = q[..., 2]
    matrices[..., 1, 1] = q[..., 3]
    matrices[..., 1, 2] = matrices[..., 2, 1] = q[..., 4]
    matrices[..., 2, 2] = -q[..., 0] - q[..., 3]
    expected = np.linalg.eigvalsh(matrices)[..., -1]

    result = _compact_q_statistics(
        path,
        shape=shape,
        dtype=np.dtype("float64"),
        s_bulk=1.0,
        ldg_a=0.0,
        ldg_b=0.0,
        ldg_c=0.0,
        chunk_x=2,
    )

    assert result["principal_S_mean"] == pytest.approx(
        float(np.mean(expected)), rel=2e-13, abs=2e-14
    )
    assert result["principal_S_std"] == pytest.approx(
        float(np.std(expected)), rel=2e-13, abs=2e-14
    )
    assert result["principal_S_min"] == pytest.approx(float(np.min(expected)))
    assert result["principal_S_max"] == pytest.approx(float(np.max(expected)))


def test_one_missing_frame_disables_acf_and_provisional_classification():
    values = [2.0] * 21
    values[5] = None
    result = summarize_series(
        np.arange(21, dtype=float) * 2.5,
        values,
        pass_relative_drift=0.1,
        fail_relative_drift=0.2,
        min_effective_samples=5,
        min_finite_frames=15,
        max_acf_lag=7,
    )

    assert result["classification"] == CLASS_INCONCLUSIVE
    assert result["finite_frame_count"] == 20
    assert result["autocorrelation"]["status"] == "not_estimated_missing_frames"
    assert result["autocorrelation"]["effective_sample_size"] is None
    assert "missing_frames_break_uniform_lag_acf" in result["reason_codes"]


def test_acf_reports_g_tau_and_effective_sample_size_consistently():
    result = stationarity._acf_summary(
        np.asarray([1.0, -1.0] * 10 + [1.0]),
        max_lag=7,
        frame_spacing=2.5,
    )

    assert result["statistical_inefficiency_frames"] == pytest.approx(1.0)
    assert result["integrated_autocorrelation_time_frames"] == pytest.approx(0.5)
    assert result["integrated_autocorrelation_time_time"] == pytest.approx(1.25)
    assert result["effective_sample_size"] == pytest.approx(21.0)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.arange(20, dtype=float), "equal lengths"),
        (
            np.asarray([*range(20), np.nan], dtype=float),
            "all be finite",
        ),
        (
            np.asarray([*range(20), 19], dtype=float),
            "strictly increasing",
        ),
        (
            np.asarray([*range(20), 21], dtype=float),
            "equally spaced",
        ),
    ],
)
def test_stationarity_summary_rejects_invalid_time_axes(times, message):
    with pytest.raises(ValueError, match=message):
        summarize_series(
            times,
            np.ones(21),
            pass_relative_drift=0.1,
            fail_relative_drift=0.2,
            min_effective_samples=5,
            min_finite_frames=15,
            max_acf_lag=7,
        )


def test_manifest_detects_finite_snapshot_replacement(tmp_path):
    kwargs = _analysis_kwargs(tmp_path)
    q_path = kwargs["run_dir"] / "Q_7.npy"
    q = np.load(q_path, allow_pickle=False)
    q[..., 1] = 0.0125
    np.save(q_path, q)

    with pytest.raises(ValueError, match="checksum manifest SHA-256 mismatch"):
        analyze(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_manifest_missing_critical_record_is_rejected(tmp_path):
    kwargs = _analysis_kwargs(tmp_path)
    manifest_path = kwargs["checksum_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"] = [
        row for row in manifest["entries"] if not row["path"].endswith("Q_9.npy")
    ]
    manifest["payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "payload_sha256"
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    kwargs["expected_checksum_manifest_sha256"] = _sha256_file(
        manifest_path
    )

    with pytest.raises(ValueError, match="missing critical inputs"):
        analyze(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_dangling_output_symlink_is_rejected_without_following_it(tmp_path):
    kwargs = _analysis_kwargs(tmp_path)
    target = tmp_path / "uncreated_target"
    link = tmp_path / "analysis_link"
    link.symlink_to(target, target_is_directory=True)
    kwargs["output_dir"] = link

    with pytest.raises(ValueError, match="must not be symlinks"):
        analyze(**kwargs)
    assert not target.exists()



def test_nematics3d_source_tree_binding_hashes_all_regular_sources(tmp_path):
    package_root = tmp_path / "nematics3d"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    (package_root / "field.py").write_text("VALUE = 2\n", encoding="utf-8")
    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "field.pyc").write_bytes(b"ignored")

    result = stationarity._bind_source_tree(package_root)

    assert result["file_count"] == 2
    assert {
        artifact["relative_path"] for artifact in result["artifacts"]
    } == {"__init__.py", "field.py"}
    assert len(result["tree_sha256"]) == 64
    for artifact in result["artifacts"]:
        assert artifact["sha256"] == _sha256_file(Path(artifact["path"]))


def test_final_rehash_detects_tamper_with_restored_file_identity(
    tmp_path, monkeypatch
):
    kwargs = _analysis_kwargs(tmp_path)
    q_path = kwargs["run_dir"] / "Q_7.npy"
    original_stat = q_path.stat()
    original_plot = stationarity._plot_png

    def mutate_after_analysis(rows, *, include_defects):
        payload = original_plot(rows, include_defects=include_defects)
        q = np.load(q_path, mmap_mode="r+", allow_pickle=False)
        q[0, 0, 0, 1] += 0.125
        q.flush()
        del q
        os.utime(
            q_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        restored = q_path.stat()
        assert restored.st_ino == original_stat.st_ino
        assert restored.st_size == original_stat.st_size
        assert restored.st_mtime_ns == original_stat.st_mtime_ns
        return payload

    monkeypatch.setattr(stationarity, "_plot_png", mutate_after_analysis)
    with pytest.raises(ValueError, match="checksum manifest SHA-256 mismatch"):
        analyze(**kwargs)
    assert not kwargs["output_dir"].exists()



def test_wrong_expected_checksum_manifest_hash_is_rejected(tmp_path):
    kwargs = _analysis_kwargs(tmp_path)
    kwargs["expected_checksum_manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checksum manifest SHA-256 mismatch"):
        analyze(**kwargs)
    assert not kwargs["output_dir"].exists()
