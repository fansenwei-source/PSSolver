import math
import json

import numpy as np
import pytest

from benchmarks.shendruk_v1_dynamic_velocity import (
    fig4_timestep_convergence as convergence,
)


def test_canonical_timestep_matrix_has_common_physical_outputs():
    rows = convergence.build_rows("0" * 40)

    assert [row["config"]["dt"] for row in rows] == [0.01, 0.005, 0.0025]
    assert [row["config"]["steps"] for row in rows] == [100, 200, 400]
    assert [row["config"]["save_start_step"] for row in rows] == [25, 50, 100]
    assert [row["config"]["save_interval"] for row in rows] == [25, 50, 100]
    assert [row["config"]["diagnostic_interval"] for row in rows] == [5, 10, 20]

    for row in rows:
        config = row["config"]
        assert config["study"] == "inertial_timestep_convergence"
        assert config["final_time"] == 1.0
        assert config["common_output_times"] == [0.25, 0.5, 0.75, 1.0]
        assert config["height"] == 20
        assert config["activity_number"] == 18.0
        assert (config["nx"], config["ny"], config["nz"]) == (320, 320, 80)
        assert config["initialization_protocol"] == "v1"
        assert config["seed"] == 24
        assert config["density"] == 1.0
        assert config["fric"] == 0.0
        assert config["mean_flow_policy"] == "evolve"
        assert config["transform_execution_order"] == "real_first"
        assert config["dtype"] == "float64"
        assert config["dealias_rule"] == "cubic_half"
        assert config["save_hydrodynamics"]
        assert row["config_sha256"] == convergence.base.canonical_sha256(config)


def test_commands_explicitly_bind_real_first_and_hydrodynamics(tmp_path):
    config = convergence.convergence_config(0.005, "1" * 40)
    config_hash = convergence.base.canonical_sha256(config)
    command, run_dir = convergence.base.command_for_config(
        config,
        config_sha256=config_hash,
        output_root=tmp_path,
        python_bin="/fixed/python",
    )

    assert run_dir == tmp_path / config["run_id"]
    assert command[command.index("--dt") + 1] == "0.0050000000000000001"
    assert command[command.index("--steps") + 1] == "200"
    assert command[command.index("--save-start-step") + 1] == "50"
    assert command[command.index("--save-interval") + 1] == "50"
    assert command[command.index("--transform-execution-order") + 1] == "real_first"
    assert "--save-hydrodynamics" in command
    assert command[command.index("--validation-config-sha256") + 1] == config_hash


def test_array_error_metrics_and_pressure_gauge_removal():
    reference = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    shifted = reference + 7.0

    raw = convergence.array_error_metrics(shifted, reference)
    demeaned = convergence.array_error_metrics(shifted, reference, demean=True)

    assert raw["max_abs"] == 7.0
    assert raw["rms"] == 7.0
    assert demeaned["max_abs"] == pytest.approx(0.0, abs=2.0e-15)
    assert demeaned["rms"] == pytest.approx(0.0, abs=2.0e-15)
    assert demeaned["first_mean_removed"] - demeaned["second_mean_removed"] == 7.0


def test_array_error_metrics_reject_layout_and_nonfinite_values():
    finite = np.zeros((2, 3), dtype=np.float64)
    nonfinite = finite.copy()
    nonfinite[0, 0] = np.nan

    with pytest.raises(ValueError, match="shape and dtype"):
        convergence.array_error_metrics(finite, np.zeros((3, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="float64"):
        convergence.array_error_metrics(
            finite.astype(np.float32), finite.astype(np.float32)
        )
    with pytest.raises(ValueError, match="finite"):
        convergence.array_error_metrics(nonfinite, finite)


def test_field_observables_have_expected_normalization():
    q = np.full((2, 2, 2, 5), 2.0, dtype=np.float64)
    u = np.zeros((2, 2, 2, 3), dtype=np.float64)
    u[..., 0] = 3.0
    u[..., 1] = 4.0
    p = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 10.0

    observed = convergence.field_observables(q, u, p, rho=2.0)

    assert observed["q_rms"] == 2.0
    assert observed["speed_rms"] == 5.0
    assert observed["kinetic_energy_density"] == 25.0
    assert observed["mean_velocity"] == [3.0, 4.0, 0.0]
    assert observed["pressure_mean_removed"] == 13.5
    assert observed["pressure_rms_demeaned"] == pytest.approx(
        math.sqrt(5.25)
    )


def test_observed_order_recovers_first_order_and_handles_roundoff_limit():
    assert convergence.observed_order(0.2, 0.1) == pytest.approx(1.0)
    assert convergence.observed_order(0.4, 0.1) == pytest.approx(2.0)
    assert convergence.observed_order(0.0, 0.0) is None
    assert convergence.observed_order(1.0, 0.0) is None
    with pytest.raises(ValueError, match="non-negative"):
        convergence.observed_order(-1.0, 0.5)


def test_noncommensurate_physical_time_is_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        convergence._integer_steps(0.3, 0.04, "test_time")


def _write_synthetic_run(output_root, config, config_hash):
    run_dir = output_root / config["run_id"]
    run_dir.mkdir(parents=True)
    metadata = {
        "status": "complete",
        "script": str(
            convergence.MODEL_SCRIPT.relative_to(convergence.PROJECT_ROOT)
        ),
        "validation_config_sha256": config_hash,
        "completed_steps": config["steps"],
        "shape": [config["nx"], config["ny"], config["nz"]],
        "dt": config["dt"],
        "density": config["density"],
        "friction": config["fric"],
        "mean_flow_policy": "evolve",
        "initialization_protocol": "v1",
        "dealias_rule": "cubic_half",
        "dtype": "float64",
        "tf32": "off",
        "transform_execution_order": "real_first",
        "solver": {"transform_execution_order": "real_first"},
        "model": {
            "variant": "beris_edwards_complete_nematic_stress_navier_stokes",
            "flow_dynamics": {
                "regime": "incompressible_navier_stokes_beris_edwards"
            },
        },
        "numerics": {
            "spectral_refresh": {"mode": "disabled"},
            "transforms": {"execution_order": "real_first"},
        },
        "initial_condition": {
            "raw_q_sha256": "a" * 64,
            "projected_q_sha256": "b" * 64,
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")

    spatial = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    for time_value in convergence.COMMON_OUTPUT_TIMES:
        step = convergence._integer_steps(time_value, config["dt"], "time")
        q = np.empty((2, 2, 2, 5), dtype=np.float64)
        for component in range(5):
            q[..., component] = (
                time_value
                + component * 0.1
                + spatial * 0.001
                + config["dt"] * 0.1
            )
        u = np.zeros((2, 2, 2, 3), dtype=np.float64)
        u[..., 0] = time_value + config["dt"] * 0.1
        u[..., 1] = 0.5 * time_value + config["dt"] * 0.05
        p = (
            time_value
            + spatial * 0.01
            + config["dt"] * 0.001 * spatial
        )
        np.save(run_dir / f"Q_{step}.npy", q)
        np.save(run_dir / f"u_{step}.npy", u)
        np.save(run_dir / f"p_{step}.npy", p)

    diagnostics = np.zeros(
        1,
        dtype=[
            ("div_max", np.float64),
            ("div_rms", np.float64),
            ("div_rel", np.float64),
            ("schur_rel_residual", np.float64),
            ("time_discrete_momentum_residual_max", np.float64),
            ("time_discrete_momentum_residual_rms", np.float64),
        ],
    )
    np.save(run_dir / "diagnostics.npy", diagnostics)


def test_synthetic_three_run_analysis_recovers_order_and_recommends_dt005(
    tmp_path,
):
    rows = []
    for dt in convergence.DT_VALUES:
        config = convergence.convergence_config(dt, "2" * 40)
        config.update({"nx": 2, "ny": 2, "nz": 2})
        config_hash = convergence.base.canonical_sha256(config)
        rows.append(
            {
                "index": len(rows),
                "config_sha256": config_hash,
                "config": config,
            }
        )
        _write_synthetic_run(tmp_path, config, config_hash)
    plan = {
        "git": {"head": "2" * 40},
        "rows": rows,
        "acceptance": convergence.build_plan()["acceptance"],
    }

    report = convergence.analyze_runs(plan, tmp_path)

    assessment = report["assessment"]
    assert report["initial_condition_identity"]["passed"]
    assert assessment["monotone_refinement_q_u"]
    assert assessment["median_observed_order_q_u_relative_l2"] == pytest.approx(
        1.0, abs=2.0e-3
    )
    assert assessment["first_order_gate_passed"]
    assert assessment["dt_0p005_short_time_adequate"]
    assert assessment["recommended_dt"] == 0.005
    assert assessment["eligible_for_inertial_fig4_pilot"]
