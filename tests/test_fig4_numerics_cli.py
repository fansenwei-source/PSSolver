import json
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "Plane_fig4_benchmark.py"


def run_dry_run(*extra_arguments):
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
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


if __name__ == "__main__":
    unittest.main()
