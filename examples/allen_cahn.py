"""A component-only Allen--Cahn example; no model-specific engine changes."""

from pssolver.core.terms import EvolvedField, Explicit, Linear, Model, Parameter, lap
from pssolver.planning.assemble import (
    AssemblyProblem,
    EulerInput,
    ExecutionInput,
    GeometryInput,
    NumericsInput,
    OutputInput,
    assemble,
)


class AllenCahn(Model):
    """∂φ/∂t = ε² ∇²φ + φ − φ³ on a dimensionless periodic box."""

    phi = EvolvedField()
    epsilon = Parameter(default=0.05)

    def terms(self, f, p):
        yield Linear(f.phi, p.epsilon**2 * lap(f.phi))
        yield Explicit(f.phi, lambda phi: phi - phi**3)


def build(*, size=8, length=1.0, dt=0.001, dtype, device="cpu", batch=1, model=None):
    """Assemble and materialize the internal periodic component interface."""
    problem = AssemblyProblem(
        AllenCahn() if model is None else model,
        GeometryInput("periodic_box", (size,) * 3, (length,) * 3),
        {"phi": ("periodic",) * 3},
        EulerInput(dt),
        NumericsInput("hermitian_half", 2, "real_first", "none", "full"),
        ExecutionInput(),
        OutputInput(("phi",)),
    )
    return assemble(problem).engine(device=device, dtype=dtype, batch=batch)


def run(initial, *, steps, dt=0.001, epsilon=0.05):
    """Run the assembled example from one explicitly supplied scalar field."""
    engine = build(
        size=initial.shape[-1],
        dt=dt,
        dtype=initial.dtype,
        device=initial.device,
        model=AllenCahn(epsilon=epsilon),
    )
    state = engine.initial_state({"phi": initial})
    for _ in range(steps):
        engine.step_(state)
    return engine, state
