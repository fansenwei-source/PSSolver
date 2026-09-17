import unittest

import torch

from pssolver.integrator import SemiImplicitEulerIntegrator


class FakeFields:
    def __init__(self):
        self.L_hat = torch.zeros((1, 1, 2), dtype=torch.float64)
        self.spatial = torch.zeros((1, 1, 2), dtype=torch.float64)
        self.spectral = torch.zeros((1, 1, 2), dtype=torch.float64)
        self.dyn_count = 1
        self.stat_count = 0
        self.forward_calls = 0
        self.inverse_calls = 0

    def group_indices_by_boundary_conditions(self, indices):
        return [list(indices)]

    def forward_transform_group(self, group):
        self.forward_calls += 1
        return self.spatial[group].clone()

    def inverse_transform_group(self, group):
        self.inverse_calls += 1
        return self.spectral[group].clone()

    def store_spatial_group(self, group, values):
        self.spatial[group] = values

    def store_spectral_group(self, group, values):
        self.spectral[group] = values


class FakeModel:
    def __init__(self):
        self.fields = FakeFields()
        self.static_updates = 0

    def update_static_fields(self):
        self.static_updates += 1

    def compute_nonlinear(self):
        return torch.zeros_like(self.fields.spectral[: self.fields.dyn_count])


class HookCountingIntegrator(SemiImplicitEulerIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_calls = 0

    def _refresh_dynamic_spectra(self):
        self.hook_calls += 1
        super()._refresh_dynamic_spectra()


def build_integrator(cls=SemiImplicitEulerIntegrator):
    model = FakeModel()
    integrator = cls(model, 0.1, None, None, None)
    return integrator, model


class SpectralRefreshTests(unittest.TestCase):
    def test_interval_rejects_invalid_values(self):
        integrator, _ = build_integrator()

        for value in (False, 0, -1, 1.5, "20"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    integrator.spectral_refresh_interval = value

    def test_scheduled_refresh_uses_overridable_hook_and_counts_events(self):
        integrator, model = build_integrator(HookCountingIntegrator)
        integrator.set_spectral_refresh_interval(2)

        integrator.step()
        self.assertEqual(integrator.step_count, 1)
        self.assertEqual(integrator.refresh_count, 0)
        self.assertEqual(integrator.hook_calls, 0)

        integrator.step()
        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.refresh_count, 1)
        self.assertEqual(integrator.hook_calls, 1)
        self.assertEqual(model.fields.forward_calls, 1)

    def test_none_disables_only_periodic_round_trip(self):
        integrator, model = build_integrator()
        integrator.spectral_refresh_interval = None

        for _ in range(5):
            integrator.step()

        self.assertEqual(integrator.step_count, 5)
        self.assertEqual(integrator.refresh_count, 0)
        self.assertEqual(model.fields.forward_calls, 0)
        self.assertEqual(model.fields.inverse_calls, 5)
        self.assertEqual(model.static_updates, 5)

    def test_restore_progress_recovers_enabled_refresh_phase_and_count(self):
        integrator, model = build_integrator()
        integrator.spectral_refresh_interval = 20
        integrator.restore_progress(45)

        self.assertEqual(integrator.step_count, 5)
        self.assertEqual(integrator.refresh_count, 2)
        for _ in range(15):
            integrator.step()

        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.refresh_count, 3)
        self.assertEqual(model.fields.forward_calls, 1)

    def test_restore_progress_with_disabled_refresh_keeps_full_step_count(self):
        integrator, model = build_integrator()
        integrator.spectral_refresh_interval = None
        integrator.restore_progress(45)
        integrator.step()

        self.assertEqual(integrator.step_count, 46)
        self.assertEqual(integrator.refresh_count, 0)
        self.assertEqual(model.fields.forward_calls, 0)


if __name__ == "__main__":
    unittest.main()
