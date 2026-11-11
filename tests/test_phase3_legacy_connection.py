"""Public export assertions retained from P3.5; retired wiring runs in history."""

import pssolver
import pssolver.integrators as integrators


def test_p35_keeps_public_facade_and_metadata_surface():
    assert "StateBackedProjectedIntegratorMixin" not in integrators.__all__
    assert not hasattr(pssolver, "StateBackedProjectedIntegratorMixin")
