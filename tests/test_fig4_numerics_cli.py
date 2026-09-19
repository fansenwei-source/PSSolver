import json
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "Plane_fig4_benchmark.py"
BERIS_EDWARDS_SCRIPT = REPOSITORY_ROOT / "Plane_beris_edwards_stokes.py"


def run_dry_run(*extra_arguments, script=BENCHMARK_SCRIPT):
    command = [
        sys.executable,
        str(script),
        "--activity-number",
        "18",
        "--output-dir",
        "/tmp/pssolver_fig4_dry_run_unused",
        "--device",
        "cpu",
        "--dry-run",
        *extra_arguments,
    ]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


class Fig4NumericsCliTests(unittest.TestCase):
    def test_default_refresh_holds_physical_interval_fixed(self):
        coarse = run_dry_run("--dt", "0.01")
        fine = run_dry_run("--dt", "0.005")
        self.assertEqual(coarse.returncode, 0, coarse.stderr)
        self.assertEqual(fine.returncode, 0, fine.stderr)

        coarse_refresh = json.loads(coarse.stdout)["numerics"]["spectral_refresh"]
        fine_refresh = json.loads(fine.stdout)["numerics"]["spectral_refresh"]
        self.assertEqual(coarse_refresh["mode"], "physical_time")
        self.assertEqual(fine_refresh["mode"], "physical_time")
        self.assertEqual(coarse_refresh["effective_interval_steps"], 20)
        self.assertEqual(fine_refresh["effective_interval_steps"], 40)
        self.assertAlmostEqual(coarse_refresh["effective_interval_time"], 0.2)
        self.assertAlmostEqual(fine_refresh["effective_interval_time"], 0.2)

    def test_refresh_modes_and_precision_are_recorded(self):
        disabled = run_dry_run(
            "--dtype",
            "float64",
            "--tf32",
            "on",
            "--disable-spectral-refresh",
        )
        legacy = run_dry_run("--spectral-refresh-steps", "7")
        self.assertEqual(disabled.returncode, 0, disabled.stderr)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)

        disabled_metadata = json.loads(disabled.stdout)
        disabled_refresh = disabled_metadata["numerics"]["spectral_refresh"]
        precision = disabled_metadata["numerics"]["precision"]
        self.assertEqual(disabled_refresh["mode"], "disabled")
        self.assertIsNone(disabled_refresh["effective_interval_steps"])
        self.assertEqual(precision["real_dtype"], "float64")
        self.assertEqual(precision["spectral_dtype"], "complex128")
        self.assertFalse(precision["tf32_effective"])

        legacy_refresh = json.loads(legacy.stdout)["numerics"]["spectral_refresh"]
        self.assertEqual(legacy_refresh["mode"], "steps")
        self.assertEqual(legacy_refresh["requested_interval_steps"], 7)
        self.assertAlmostEqual(legacy_refresh["effective_interval_time"], 0.07)

    def test_physical_refresh_rejects_noninteger_dt_ratio(self):
        result = run_dry_run("--dt", "0.03")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integer multiple", result.stderr)


    def test_beris_edwards_identity_parameters_and_hashes_are_recorded(self):
        result = run_dry_run(
            "--dtype",
            "float64",
            "--disable-spectral-refresh",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        model = metadata["model"]
        parameters = model["parameters"]
        self.assertEqual(metadata["script"], "Plane_beris_edwards_stokes.py")
        runtime = metadata["runtime_environment"]
        self.assertEqual(runtime["device_type"], "cpu")
        self.assertEqual(runtime["resolved_device"], "cpu")
        self.assertIsNone(runtime["cuda_device_name"])
        self.assertIsInstance(runtime["python_version"], str)
        self.assertIsInstance(runtime["torch_version"], str)
        self.assertEqual(
            model["variant"],
            "beris_edwards_complete_nematic_stress_stokes",
        )
        self.assertEqual(
            model["q_dynamics"]["implementation"],
            "pssolver.models.active_nematics.BerisEdwardsQNonlinearModel",
        )
        self.assertIn("complete_one_constant", model["flow_dynamics"]["nematic_stress"])
        self.assertAlmostEqual(parameters["activity_number"], 18.0)
        self.assertAlmostEqual(parameters["zeta"], 0.01)
        self.assertAlmostEqual(parameters["frank_K"], 0.012345679012345678)
        self.assertAlmostEqual(parameters["ldg_L1"], 0.024691358024691357)
        self.assertAlmostEqual(parameters["rotational_viscosity_gamma"], 2.94)
        self.assertAlmostEqual(parameters["flow_alignment_lambda"], 0.3)
        self.assertEqual(parameters["fric"], 0.0)
        self.assertEqual(metadata["numerics"]["spectral_refresh"]["mode"], "disabled")
        self.assertEqual(metadata["numerics"]["precision"]["real_dtype"], "float64")
        self.assertTrue(metadata["numerics"]["q_gradient_reuse"]["enabled"])
        self.assertEqual(
            metadata["numerics"]["molecular_field_linear_space"],
            "spectral",
        )
        self.assertEqual(
            metadata["numerics"]["stress_divergence_sum_space"],
            "spectral",
        )
        self.assertEqual(metadata["pointwise_execution"], "compile")
        self.assertEqual(
            metadata["model"]["q_dynamics"]["pointwise_execution"],
            "compile",
        )
        self.assertEqual(
            metadata["model"]["flow_dynamics"]["pointwise_execution"],
            "compile",
        )
        pointwise = metadata["numerics"]["pointwise_kernels"]
        self.assertEqual(pointwise["requested"], "compile")
        self.assertEqual(pointwise["effective"], "compile")
        self.assertFalse(pointwise["fallback_allowed"])
        self.assertTrue(pointwise["compile"]["enabled"])
        projected = metadata["numerics"]["dealiasing"][
            "projected_transform_execution"
        ]
        self.assertEqual(projected["requested"], "truncated")
        self.assertEqual(projected["effective"], "truncated")
        self.assertFalse(projected["fallback_allowed"])
        self.assertTrue(projected["truncated_real_basis_axes"])
        self.assertFalse(projected["full_spectral_storage_preserved"])
        self.assertEqual(
            metadata["numerics"]["transforms"]["execution_order"],
            "real_first",
        )
        self.assertIsNone(metadata["validation_config_sha256"])

        files = metadata["implementation_provenance"]["files"]
        self.assertEqual(
            set(files),
            {
                "Plane_beris_edwards_stokes.py",
                "pssolver/applications/__init__.py",
                "pssolver/applications/plane_beris_edwards.py",
                "pssolver/solver.py",
                "pssolver/Field.py",
                "pssolver/PDEmodel.py",
                "pssolver/integrator.py",
                "pssolver/transforms.py",
                "pssolver/backends/__init__.py",
                "pssolver/backends/bounded.py",
                "pssolver/backends/tensor_product.py",
                "pssolver/plane.py",
                "pssolver/__init__.py",
                "pssolver/adapters/legacy_boundaries.py",
                "pssolver/configuration/__init__.py",
                "pssolver/configuration/plane_beris_edwards.py",
                "pssolver/core/boundary.py",
                "pssolver/core/domain.py",
                "pssolver/core/geometry.py",
                "pssolver/core/numerics.py",
                "pssolver/geometries/tensor_product.py",
                "pssolver/models/active_nematics/__init__.py",
                "pssolver/models/active_nematics/fields.py",
                "pssolver/models/active_nematics/q_tensor.py",
                "pssolver/models/active_nematics/beris_edwards.py",
                "pssolver/models/active_nematics/stokes.py",
                "pssolver/models/active_nematics/initial_conditions.py",
                "pssolver/presets/__init__.py",
                "pssolver/presets/shendruk.py",
                "pssolver/runtime/__init__.py",
                "pssolver/runtime/plane_beris_edwards.py",
                "pssolver/runtime/plane_legacy.py",
                "pssolver/workflows/__init__.py",
                "pssolver/workflows/plane_checkpoint.py",
                "pssolver/workflows/plane_observation.py",
                "pssolver/workflows/plane_beris_edwards.py",
            },
        )
        self.assertTrue(
            all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in files.values()
            )
        )

    def test_beris_edwards_validation_config_sha_is_recorded_and_validated(self):
        digest = "a" * 64
        valid = run_dry_run(
            "--validation-config-sha256",
            digest,
            script=BERIS_EDWARDS_SCRIPT,
        )
        invalid = run_dry_run(
            "--validation-config-sha256",
            "A" * 64,
            script=BERIS_EDWARDS_SCRIPT,
        )

        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(
            json.loads(valid.stdout)["validation_config_sha256"],
            digest,
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("64 lowercase hexadecimal", invalid.stderr)

    def test_beris_edwards_friction_mode_records_positive_drag(self):
        result = run_dry_run(
            "--zero-mode-policy",
            "friction",
            "--friction-mode-fric",
            "0.125",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["model"]["parameters"]["fric"], 0.125)
        self.assertEqual(metadata["numerics"]["velocity_zero_mode"], "friction")

    def test_beris_edwards_q_gradient_reuse_can_be_disabled(self):
        result = run_dry_run(
            "--disable-q-gradient-reuse",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        reuse = json.loads(result.stdout)["numerics"]["q_gradient_reuse"]
        self.assertFalse(reuse["enabled"])
        self.assertEqual(reuse["scope"], "single_static_to_nonlinear_evaluation")

    def test_beris_edwards_real_first_transform_order_is_recorded(self):
        result = run_dry_run(
            "--transform-execution-order",
            "real_first",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        transforms = metadata["numerics"]["transforms"]
        self.assertEqual(transforms["execution_order"], "real_first")
        self.assertEqual(transforms["spectral_storage"], "hermitian_half")
        self.assertEqual(transforms["hermitian_axis"], 1)
        self.assertFalse(transforms["basis_and_normalization_changed"])
        self.assertEqual(metadata["transform_execution_order"], "real_first")

    def test_beris_edwards_hermitian_half_storage_is_explicit_and_recorded(self):
        result = run_dry_run(
            "--nx",
            "12",
            "--ny",
            "10",
            "--nz",
            "8",
            "--spectral-storage",
            "hermitian_half",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        transforms = metadata["numerics"]["transforms"]
        self.assertEqual(transforms["spectral_storage"], "hermitian_half")
        self.assertEqual(transforms["physical_shape"], [12, 10, 8])
        self.assertEqual(transforms["spectral_shape"], [12, 6, 8])
        self.assertEqual(transforms["hermitian_axis"], 1)
        self.assertFalse(transforms["basis_and_normalization_changed"])
        self.assertEqual(metadata["solver"]["spectral_shape"], [12, 6, 8])
        self.assertEqual(metadata["spectral_storage"], "hermitian_half")
        projected = metadata["numerics"]["dealiasing"][
            "projected_transform_execution"
        ]
        self.assertEqual(projected["spectral_storage"], "hermitian_half")
        self.assertTrue(projected["backend_storage_shape_preserved"])
        self.assertFalse(projected["full_spectral_storage_preserved"])

    def test_beris_edwards_hermitian_half_rejects_legacy_order(self):
        result = run_dry_run(
            "--spectral-storage",
            "hermitian_half",
            "--transform-execution-order",
            "legacy",
            script=BERIS_EDWARDS_SCRIPT,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires", result.stderr)

    def test_beris_edwards_physical_linear_molecular_field_fallback_is_recorded(self):
        result = run_dry_run(
            "--molecular-field-linear-space",
            "physical",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        self.assertEqual(
            metadata["numerics"]["molecular_field_linear_space"],
            "physical",
        )
        self.assertEqual(
            metadata["model"]["flow_dynamics"][
                "molecular_field_linear_space"
            ],
            "physical",
        )
        self.assertEqual(
            metadata["molecular_field_linear_space"],
            "physical",
        )

    def test_beris_edwards_physical_stress_divergence_sum_is_recorded(self):
        result = run_dry_run(
            "--stress-divergence-sum-space",
            "physical",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        self.assertEqual(
            metadata["numerics"]["stress_divergence_sum_space"],
            "physical",
        )
        self.assertEqual(
            metadata["model"]["flow_dynamics"][
                "stress_divergence_sum_space"
            ],
            "physical",
        )
        self.assertEqual(
            metadata["stress_divergence_sum_space"],
            "physical",
        )

    def test_beris_edwards_eager_pointwise_rollback_is_recorded(self):
        result = run_dry_run(
            "--pointwise-execution",
            "eager",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["pointwise_execution"], "eager")
        self.assertEqual(
            metadata["model"]["q_dynamics"]["pointwise_execution"],
            "eager",
        )
        self.assertEqual(
            metadata["model"]["flow_dynamics"]["pointwise_execution"],
            "eager",
        )
        pointwise = metadata["numerics"]["pointwise_kernels"]
        self.assertEqual(pointwise["requested"], "eager")
        self.assertEqual(pointwise["effective"], "eager")
        self.assertFalse(pointwise["fallback_allowed"])
        self.assertFalse(pointwise["compile"]["enabled"])
        self.assertIsNone(pointwise["compile"]["backend"])
        self.assertIsNone(pointwise["compile"]["mode"])
        self.assertIsNone(pointwise["compile"]["dynamic"])
        self.assertIsNone(pointwise["compile"]["fullgraph"])

    def test_beris_edwards_truncated_projected_transforms_are_recorded(self):
        result = run_dry_run(
            "--projected-transform-execution",
            "truncated",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        projected = metadata["numerics"]["dealiasing"][
            "projected_transform_execution"
        ]
        self.assertEqual(projected["requested"], "truncated")
        self.assertEqual(projected["effective"], "truncated")
        self.assertFalse(projected["fallback_allowed"])
        self.assertTrue(projected["truncated_real_basis_axes"])
        self.assertFalse(projected["full_spectral_storage_preserved"])
        self.assertEqual(
            metadata["projected_transform_execution"],
            "truncated",
        )

    def test_beris_edwards_full_projected_transform_rollback_is_recorded(self):
        result = run_dry_run(
            "--projected-transform-execution",
            "full",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        projected = metadata["numerics"]["dealiasing"][
            "projected_transform_execution"
        ]
        self.assertEqual(projected["requested"], "full")
        self.assertEqual(projected["effective"], "full")
        self.assertFalse(projected["fallback_allowed"])
        self.assertFalse(projected["truncated_real_basis_axes"])
        self.assertEqual(metadata["projected_transform_execution"], "full")

    def test_truncated_projected_transforms_reject_disabled_dealiasing(self):
        result = run_dry_run(
            "--dealias-rule",
            "none",
            "--projected-transform-execution",
            "truncated",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires enabled dealiasing", result.stderr)

    def test_beris_edwards_legacy_transform_order_remains_available(self):
        result = run_dry_run(
            "--transform-execution-order",
            "legacy",
            "--spectral-storage",
            "full_complex",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        transforms = metadata["numerics"]["transforms"]
        self.assertEqual(transforms["execution_order"], "legacy")
        self.assertEqual(transforms["spectral_storage"], "full_complex")
        self.assertEqual(metadata["transform_execution_order"], "legacy")

    def test_beris_edwards_full_complex_storage_rollback_is_recorded(self):
        result = run_dry_run(
            "--spectral-storage",
            "full_complex",
            script=BERIS_EDWARDS_SCRIPT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        metadata = json.loads(result.stdout)
        transforms = metadata["numerics"]["transforms"]
        self.assertEqual(transforms["spectral_storage"], "full_complex")
        self.assertEqual(transforms["spectral_shape"], transforms["physical_shape"])
        self.assertIsNone(transforms["hermitian_axis"])
        self.assertEqual(metadata["spectral_storage"], "full_complex")


if __name__ == "__main__":
    unittest.main()
