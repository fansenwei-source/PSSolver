import json
import numpy as np
from pathlib import Path
import tempfile
import torch
import unittest

from pssolver import (
    SpectralSolver,
    apply_snapshot_to_solver,
    load_snapshot,
    require_distinct_output_directory,
)
from pssolver.models.active_nematics import Q_convention_metadata
from pssolver.snapshots import (
    load_q_snapshot,
    representative_ordered_S,
)


Q_NAMES = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
U_NAMES = ("ux", "uy", "uz")


def save_snapshot(
    directory,
    step,
    shape=(4, 3, 2),
    *,
    model_name="active_nematics",
    convention=None,
    S_initial=0.4,
    S_bulk=0.4,
    include_S_initial=True,
    include_S_bulk=True,
    write_metadata=True,
):
    q_values = np.arange(np.prod((*shape, 5)), dtype=np.float32).reshape(*shape, 5)
    u_values = np.arange(np.prod((*shape, 3)), dtype=np.float32).reshape(*shape, 3)
    p_values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.save(directory / f"Q_{step}.npy", q_values)
    np.save(directory / f"u_{step}.npy", u_values)
    np.save(directory / f"p_{step}.npy", p_values)
    if write_metadata:
        if convention is None:
            convention = Q_convention_metadata()
        parameters = {}
        if include_S_initial:
            parameters["S_initial"] = S_initial
        if include_S_bulk:
            parameters["S_bulk"] = S_bulk
        metadata = {
            "schema_version": 1,
            "script": "snapshot_test.py",
            "solver": {
                "shape": list(shape),
                "lengths": [float(value) for value in shape],
                "dt": 0.01,
            },
            "model": {
                "name": model_name,
                "Q_convention": convention,
                "parameters": parameters,
            },
            "boundary_conditions": {
                "Q": ["periodic", "periodic", "periodic"],
                "velocity": ["periodic", "periodic", "periodic"],
                "pressure": ["periodic", "periodic", "periodic"],
            },
            "numerics": {
                "dealiasing": "none",
                "velocity_zero_mode": "periodic",
                "pressure_solver": "test",
            },
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
    return q_values, u_values, p_values


class ZeroQModel(torch.nn.Module):
    def forward(self, fields, params):
        del params
        values = torch.zeros(
            (len(Q_NAMES), fields.batchsize, *fields.shape),
            device=fields.spatial.device,
            dtype=fields.spatial.dtype,
        )
        return fields.transform_tensor(values, ("periodic",) * 3)


class CountingStaticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.pressure_guess = None
        self.last_pressure_hat = None

    def forward(self, fields, params):
        del params
        self.call_count += 1
        qxx = fields["Qxx"]
        values = torch.stack((2 * qxx, 3 * qxx, 4 * qxx, 5 * qxx))
        return fields.transform_tensor(values, ("periodic",) * 3)


def build_snapshot_solver(snapshot):
    solver = SpectralSolver(
        shape=snapshot.shape,
        L=tuple(float(size) for size in snapshot.shape),
        dt=0.01,
        device="cpu",
    )
    q2 = solver.get_q2(("periodic",) * 3)
    for name in Q_NAMES:
        solver.model.add_dynamic_field(
            name,
            init=snapshot.q_fields[name],
            L_hat=torch.zeros_like(q2),
            boundary_conditions=("periodic",) * 3,
        )
    for name in (*U_NAMES, "p"):
        solver.model.add_static_field(name, boundary_conditions=("periodic",) * 3)
    static_model = CountingStaticModel()
    solver.model.set_nonlinear_model(ZeroQModel())
    solver.model.set_static_compute_model(static_model)
    solver.build()
    return solver, static_model


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.snapshot_directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_representative_ordered_S_uses_current_Q_not_source_metadata(self):
        q = np.zeros((4, 3, 2, 5), dtype=np.float32)
        q[..., 0] = 0.4
        q[..., 3] = -0.2

        measured = representative_ordered_S(q)

        self.assertAlmostEqual(measured, 0.4, places=6)
        self.assertNotAlmostEqual(measured, 2.0 / 3.0, places=3)

    def test_representative_ordered_S_rejects_unordered_Q_for_branch_metadata(self):
        q = np.zeros((4, 3, 2, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            representative_ordered_S(q)

    def test_load_snapshot_validates_and_splits_fields(self):
        q_values, u_values, p_values = save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(
            self.snapshot_directory,
            7,
            expected_shape=(4, 3, 2),
        )

        self.assertEqual(snapshot.step, 7)
        self.assertEqual(snapshot.shape, (4, 3, 2))
        for component, name in enumerate(Q_NAMES):
            torch.testing.assert_close(
                snapshot.q_fields[name], torch.from_numpy(q_values[..., component])
            )
        for component, name in enumerate(U_NAMES):
            torch.testing.assert_close(
                snapshot.velocity_fields[name], torch.from_numpy(u_values[..., component])
            )
        torch.testing.assert_close(snapshot.pressure, torch.from_numpy(p_values))

    def test_load_snapshot_rejects_mismatched_shape(self):
        save_snapshot(self.snapshot_directory, 7)
        with self.assertRaisesRegex(ValueError, "does not match solver shape"):
            load_snapshot(
                self.snapshot_directory,
                7,
                expected_shape=(5, 3, 2),
            )

    def test_resume_uses_restored_static_fields_once(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        solver, static_model = build_snapshot_solver(snapshot)

        start_step = apply_snapshot_to_solver(
            solver,
            snapshot,
            mode="resume",
            current_run_metadata=snapshot.source_metadata.metadata,
            validate_resume_static=False,
        )
        self.assertEqual(start_step, 7)
        self.assertEqual(solver.integrator.step_count, 7)
        calls_after_build = static_model.call_count

        captured = {}
        solver.run(
            1,
            pre_update_callback=lambda current_solver, _: captured.update(
                ux=current_solver.model.fields["ux"].detach().clone(),
                p=current_solver.model.fields["p"].detach().clone(),
            ),
        )
        self.assertEqual(static_model.call_count, calls_after_build)
        torch.testing.assert_close(
            captured["ux"][0], snapshot.velocity_fields["ux"]
        )
        torch.testing.assert_close(captured["p"][0], snapshot.pressure)
        self.assertEqual(solver.integrator.step_count, 8)

        solver.run(1)
        self.assertEqual(static_model.call_count, calls_after_build + 1)

    def test_branch_recomputes_static_fields_before_first_update(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        solver, static_model = build_snapshot_solver(snapshot)

        start_step = apply_snapshot_to_solver(solver, snapshot, mode="branch")
        self.assertEqual(start_step, 0)
        calls_after_build = static_model.call_count
        solver.run(1)
        self.assertEqual(static_model.call_count, calls_after_build + 1)
        torch.testing.assert_close(
            solver.model.fields["ux"],
            2 * solver.model.fields["Qxx"],
        )

    def test_resume_rejects_static_fields_from_different_parameters(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        solver, _ = build_snapshot_solver(snapshot)

        with self.assertRaisesRegex(ValueError, "Use mode='branch'"):
            apply_snapshot_to_solver(
                solver,
                snapshot,
                mode="resume",
                current_run_metadata=snapshot.source_metadata.metadata,
            )

    def test_snapshot_output_must_not_overwrite_source(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        with self.assertRaisesRegex(ValueError, "must differ"):
            require_distinct_output_directory(snapshot, self.snapshot_directory)


    def test_loaded_snapshot_exposes_validated_provenance(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        self.assertEqual(snapshot.source_metadata.S_bulk, 0.4)
        self.assertEqual(
            snapshot.source_metadata.metadata["model"]["Q_convention"],
            Q_convention_metadata(),
        )

    def test_load_snapshot_requires_metadata_without_legacy_fallback(self):
        save_snapshot(self.snapshot_directory, 7, write_metadata=False)
        with self.assertRaisesRegex(
            FileNotFoundError,
            "Canonical active-nematic Q metadata",
        ):
            load_snapshot(self.snapshot_directory, 7)

    def test_load_snapshot_requires_known_schema_version(self):
        save_snapshot(self.snapshot_directory, 7)
        metadata_path = self.snapshot_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())

        metadata.pop("schema_version")
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "schema_version=1"):
            load_snapshot(self.snapshot_directory, 7)

        metadata["schema_version"] = 2
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "schema_version=1"):
            load_snapshot(self.snapshot_directory, 7)

    def test_load_snapshot_rejects_wrong_model_and_convention(self):
        save_snapshot(self.snapshot_directory, 7, model_name="another_model")
        with self.assertRaisesRegex(ValueError, "model.name='active_nematics'"):
            load_snapshot(self.snapshot_directory, 7)

        wrong_convention = Q_convention_metadata()
        wrong_convention["definition"] = "Q=S(nn-I/3)"
        save_snapshot(
            self.snapshot_directory,
            7,
            convention=wrong_convention,
        )
        with self.assertRaisesRegex(ValueError, "complete canonical Q convention"):
            load_snapshot(self.snapshot_directory, 7)

    def test_load_snapshot_requires_declared_S_bulk(self):
        save_snapshot(self.snapshot_directory, 7, include_S_bulk=False)
        with self.assertRaisesRegex(ValueError, "model.parameters.S_bulk"):
            load_snapshot(self.snapshot_directory, 7)

        for invalid_S_bulk in (0.0, -0.1, np.inf):
            with self.subTest(S_bulk=invalid_S_bulk):
                save_snapshot(
                    self.snapshot_directory,
                    7,
                    S_bulk=invalid_S_bulk,
                )
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    load_snapshot(self.snapshot_directory, 7)

    def test_nullable_S_bulk_and_expected_model_comparison(self):
        save_snapshot(self.snapshot_directory, 7, S_bulk=None)
        snapshot = load_snapshot(
            self.snapshot_directory,
            7,
            expected_S_bulk=None,
        )
        self.assertIsNone(snapshot.source_metadata.S_bulk)
        with self.assertRaisesRegex(ValueError, "current model requires S_bulk"):
            load_snapshot(
                self.snapshot_directory,
                7,
                expected_S_bulk=0.4,
            )

        save_snapshot(self.snapshot_directory, 7, S_bulk=0.4)
        with self.assertRaisesRegex(ValueError, "current model requires S_bulk"):
            load_snapshot(
                self.snapshot_directory,
                7,
                expected_S_bulk=0.5,
            )

    def test_q_only_loader_rejects_metadata_from_another_directory(self):
        q_values, _, _ = save_snapshot(self.snapshot_directory, 7)
        q_directory = self.snapshot_directory / "q_only"
        q_directory.mkdir()
        q_path = q_directory / "Q_7.npy"
        np.save(q_path, q_values)

        with self.assertRaisesRegex(ValueError, "must be colocated"):
            load_q_snapshot(
                q_path,
                expected_shape=(4, 3, 2),
                metadata_directory=self.snapshot_directory,
                expected_S_bulk=0.4,
                require_S_initial=True,
            )


    def test_resume_requires_current_metadata_and_rejects_changed_dt(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        solver, _ = build_snapshot_solver(snapshot)

        with self.assertRaisesRegex(ValueError, "current_run_metadata"):
            apply_snapshot_to_solver(
                solver,
                snapshot,
                mode="resume",
                validate_resume_static=False,
            )

        changed = json.loads(json.dumps(snapshot.source_metadata.metadata))
        changed["solver"]["dt"] = 0.02
        with self.assertRaisesRegex(ValueError, "trajectory-defining metadata"):
            apply_snapshot_to_solver(
                solver,
                snapshot,
                mode="resume",
                current_run_metadata=changed,
                validate_resume_static=False,
            )

    def test_resume_allows_changed_final_steps_and_save_cadence(self):
        save_snapshot(self.snapshot_directory, 7)
        snapshot = load_snapshot(self.snapshot_directory, 7)
        solver, _ = build_snapshot_solver(snapshot)
        current = json.loads(json.dumps(snapshot.source_metadata.metadata))
        current["solver"]["steps"] = 100
        current["solver"]["save_interval"] = 10

        start_step = apply_snapshot_to_solver(
            solver,
            snapshot,
            mode="resume",
            current_run_metadata=current,
            validate_resume_static=False,
        )

        self.assertEqual(start_step, 7)


if __name__ == "__main__":
    unittest.main()
