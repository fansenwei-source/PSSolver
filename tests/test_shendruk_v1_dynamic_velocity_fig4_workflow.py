import json
from pathlib import Path

import numpy as np

from benchmarks.shendruk_v1_dynamic_velocity import fig4_workflow as workflow


def test_canonical_matrix_matches_archived_v1_preview():
    rows = workflow.build_rows("0" * 40)

    assert len(rows) == 23
    observed = {
        height: (
            {row["config"]["nz"] for row in rows if row["config"]["height"] == height},
            tuple(
                row["config"]["activity_number"]
                for row in rows
                if row["config"]["height"] == height
            ),
        )
        for height in workflow.ACTIVITIES
    }
    assert observed == {
        10: ({40}, (5.0, 10.0, 15.0, 17.0, 18.0, 20.0, 22.0)),
        15: ({60}, (7.0, 12.0, 16.0, 17.0, 18.0, 20.0, 25.0, 33.0)),
        20: ({80}, (9.0, 15.0, 17.0, 18.0, 20.0, 25.0, 35.0, 44.7)),
    }

    for row in rows:
        config = row["config"]
        assert row["config_sha256"] == workflow.canonical_sha256(config)
        assert config["initialization_protocol"] == "v1"
        assert config["nx"] == config["ny"] == 320
        assert config["dt"] == 0.005
        assert config["steps"] == 40_000
        assert config["save_start_step"] == 20_000
        assert config["save_interval"] == 1_000
        assert config["seed"] == 24
        assert config["dealias_rule"] == "cubic_half"
        assert config["dtype"] == "float64"
        assert config["spectral_dtype"] == "complex128"
        assert config["tf32"] == "off"
        assert config["spectral_refresh"] == "disabled"
        assert config["density"] == 1.0
        assert config["fric"] == 0.0
        assert config["mean_flow_policy"] == "evolve"
        assert not config["save_hydrodynamics"]


def test_preflight_command_changes_only_execution_extent_and_outputs(tmp_path):
    plan = {"rows": workflow.build_rows("1" * 40)}
    config, config_sha256 = workflow.selected_config(
        plan, index=0, preflight=True
    )
    command, run_dir = workflow.command_for_config(
        config,
        config_sha256=config_sha256,
        output_root=tmp_path,
        python_bin="/fixed/python",
    )

    assert config["height"] == 20
    assert config["activity_number"] == 18.0
    assert config["nz"] == 80
    assert config["steps"] == 1
    assert config["save_start_step"] == 1
    assert config["save_hydrodynamics"]
    assert run_dir == tmp_path / config["run_id"]
    assert command[0] == "/fixed/python"
    assert str(workflow.MODEL_SCRIPT) == command[1]
    assert "--mean-flow-policy" in command
    assert command[command.index("--mean-flow-policy") + 1] == "evolve"
    assert "--density" in command
    assert command[command.index("--density") + 1] == "1"
    assert "--fric" in command
    assert command[command.index("--fric") + 1] == "0"
    assert "--disable-spectral-refresh" in command
    assert "--save-hydrodynamics" in command
    assert command[command.index("--validation-config-sha256") + 1] == (
        config_sha256
    )


def test_synthetic_preflight_output_passes_validation(tmp_path):
    config = workflow.scientific_config(
        height=4,
        nz=4,
        activity_number=18.0,
        commit="2" * 40,
    )
    config.update(
        {
            "run_id": "synthetic_preflight",
            "nx": 2,
            "ny": 3,
            "dt": 0.002,
            "steps": 1,
            "save_start_step": 1,
            "save_interval": 1,
            "diagnostic_interval": 1,
            "save_hydrodynamics": True,
        }
    )
    config_sha256 = workflow.canonical_sha256(config)
    run_dir = tmp_path / config["run_id"]
    run_dir.mkdir()

    metadata = {
        "status": "complete",
        "script": str(workflow.MODEL_SCRIPT.relative_to(workflow.PROJECT_ROOT)),
        "validation_config_sha256": config_sha256,
        "completed_steps": 1,
        "shape": [2, 3, 4],
        "dt": config["dt"],
        "density": config["density"],
        "friction": config["fric"],
        "mean_flow_policy": "evolve",
        "initialization_protocol": "v1",
        "dealias_rule": "cubic_half",
        "dtype": "float64",
        "tf32": "off",
        "model": {
            "variant": "beris_edwards_complete_nematic_stress_navier_stokes",
            "flow_dynamics": {
                "regime": "incompressible_navier_stokes_beris_edwards"
            },
        },
        "numerics": {"spectral_refresh": {"mode": "disabled"}},
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    np.save(run_dir / "Q_1.npy", np.zeros((2, 3, 4, 5), dtype=np.float64))
    np.save(run_dir / "u_1.npy", np.zeros((2, 3, 4, 3), dtype=np.float64))
    np.save(run_dir / "p_1.npy", np.zeros((2, 3, 4), dtype=np.float64))
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

    report = workflow.validate_run(run_dir, config, config_sha256)

    assert report["validation"] == "passed"
    assert [item["path"] for item in report["q_inventory"]] == ["Q_1.npy"]
    assert {item["path"] for item in report["hydrodynamic_inventory"]} == {
        "u_1.npy",
        "p_1.npy",
    }


def test_finite_scan_covers_every_chunk():
    array = np.zeros((3, 2, 2), dtype=np.float64)
    array[-1, -1, -1] = np.nan
    assert not workflow.array_is_finite(array)
