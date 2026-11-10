"""Five-step model skeleton; copy this into an application, not the engine."""

from pssolver.core.terms import EvolvedField, Explicit, Linear, Model, Parameter, lap


class MyModel(Model):
    """TODO 1: state the PDE, units, boundary and conserved quantities."""

    # TODO 2: declare every evolved/constrained field and fixed/control parameter.
    phi = EvolvedField()
    diffusivity = Parameter(default=1.0)

    def terms(self, f, p):
        """TODO 3: split a uniform diagonal linear term from physical kernels."""
        yield Linear(f.phi, p.diffusivity * lap(f.phi))
        yield Explicit(f.phi, lambda phi: phi * 0)


# TODO 4: compose AssemblyProblem (geometry, boundaries, numerics, Euler, outputs).
# TODO 5: run checks.py's MMS/convergence/invariant/restart/Taylor checks before use.
