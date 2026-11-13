"""Test-only schemes, models, and matrix-free reference implementations."""

from dataclasses import fields, replace
from math import pi

import torch

from pssolver.core.terms import EvolvedField, Explicit, Linear, Model, Time, lap
from pssolver.engine.engine import Engine
from pssolver.engine.imex import SemiImplicitEuler
from pssolver.engine.linear import ModalBlockOperator, SolveReport
from pssolver.planning.assemble import (
    AssemblyProblem,
    EulerInput,
    ExecutionInput,
    GeometryInput,
    NumericsInput,
    OutputInput,
    assemble,
)
from pssolver.planning.capabilities import implementations
from pssolver.planning.engine_plan import build_plan


class ReferenceSBDF2:
    """One-slot BDF2 with a semi-implicit Euler startup."""

    history_length = 1

    def __init__(self, dt):
        self.dt = dt

    def implicit_shift(self, history, dt):
        if history and dt != self.dt:
            raise ValueError("SBDF2 history requires unchanged dt")
        return 1.5 if history else 1.0

    def implicit_denominator(self, linear, dt):
        return 1.5 - linear * dt

    def explicit_update(self, u, nonlinear, history, dt, *, out=None):
        if not history:
            return SemiImplicitEuler().explicit_update(u, nonlinear, (), dt, out=out)
        previous, previous_rhs = history[0].unbind(0)
        new = 2 * u - .5 * previous + dt * (2 * nonlinear - previous_rhs)
        return new if out is None else out.copy_(new)

    def next_history(self, u, nonlinear, history):
        return (torch.stack((u, nonlinear)),)


class _ScalarModel(Model):
    u = EvolvedField()

    def __init__(self, diffusion, explicit):
        super().__init__()
        self.diffusion = diffusion
        self.explicit = explicit

    def terms(self, f, p):
        yield Linear(f.u, self.diffusion * lap(f.u))
        yield Explicit(f.u, self.explicit, inputs=(f.u, Time()))


def assemble_periodic(model, dt, shape):
    return assemble(AssemblyProblem(
        model, GeometryInput("periodic_box", shape, (2 * pi,) * 3),
        {"u": ("periodic",) * 3}, EulerInput(dt),
        NumericsInput("hermitian_half", 2, "real_first", "none", "full"),
        ExecutionInput(), OutputInput(("u",)),
    ))


def reference_engine(assembled, dtype, sbdf2, *, tier="diagonal", factory=None):
    basis = assembled.plan
    members = {f.name: getattr(basis, f.name) for f in fields(basis)
               if f.name != "identity_sha256"}
    if sbdf2:
        members["integrator"] = replace(basis.integrator, scheme="sbdf2_test", history_length=1)
    if tier != "diagonal":
        members["linear"] = tuple(replace(term, tier=tier) for term in basis.linear)
    plan = build_plan(**members)
    scheme = ReferenceSBDF2(basis.integrator.dt) if sbdf2 else SemiImplicitEuler()
    if factory is None:
        by_recipe = {item.implementation: item for item in implementations("linear_recipe")}
        capabilities = [by_recipe[item.recipe] for item in plan.linear]

        def factory(engine):
            return engine._generic_linear(capabilities)
    return Engine._for_reference(
        plan, device="cpu", dtype=dtype, scheme=scheme, linear_factory=factory,
        kernels=assembled.kernels,
    )


def scalar_engine(dt, dtype, sbdf2, *, shape=(32, 8, 8)):
    def explicit(u, _time):
        return .15 * u

    return reference_engine(assemble_periodic(_ScalarModel(.2, explicit), dt, shape), dtype, sbdf2)


def block_engine(dt, dtype, sbdf2, coupling, diffusion):
    class Coupled(Model):
        u = EvolvedField(tuple(f"u{i}" for i in range(len(diffusion))))

        def terms(self, f, p):
            yield Linear(f.u, lap(f.u))

    def factory(engine):
        q2 = engine.transform_backend.get_q2(engine.boundaries)
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        c = torch.tensor(coupling, dtype=cdtype)
        d = torch.diag(torch.tensor(diffusion, dtype=dtype))
        return ModalBlockOperator((c - q2[..., None, None] * d).unsqueeze(0))

    return reference_engine(
        assemble_periodic(Coupled(), dt, (32, 8, 8)), dtype, sbdf2,
        tier="modal_block_test", factory=factory,
    )


class VariableDiffusion:
    """Test-only real-space CG adapter for a periodic Hermitian spectrum."""

    def __init__(self, engine, *, maxiter=128, precondition=None):
        self.engine = engine
        n = engine.shape[0]
        x = torch.arange(n, dtype=engine.dtype) * (2 * pi / n)
        self.a = (1 + .25 * x.cos()).reshape(n, 1, 1)
        wave = torch.fft.fftfreq(n, d=1 / n, dtype=engine.dtype)
        wave[n // 2] = 0
        self.wave = (1j * wave).reshape(n, 1, 1)
        self.maxiter = maxiter
        self.preconditioner = precondition

    def derivative(self, value):
        return torch.fft.ifft(
            self.wave * torch.fft.fft(value, dim=-3), dim=-3,
        ).real

    def apply(self, value):
        return self.derivative(self.a * self.derivative(value))

    def precondition(self, residual):
        return residual if self.preconditioner is None else self.preconditioner(residual)

    def iterate(self, rhs, *, shift, dt, precondition, record):
        norm = float(torch.linalg.vector_norm(rhs))
        threshold = 1e-13 * norm
        x, r = torch.zeros_like(rhs), rhs
        residuals = [norm] if record else []
        if norm == 0:
            return x, SolveReport(True, 0, tuple(residuals))
        z = r if precondition is None else precondition(r)
        p, rz = z, torch.sum(r * z)
        for i in range(1, self.maxiter + 1):
            action = shift * p - dt * self.apply(p)
            alpha = rz / torch.sum(p * action)
            x, r = x + alpha * p, r - alpha * action
            # Check the actual residual, not only the accumulated recurrence.
            actual = float(torch.linalg.vector_norm(rhs - (shift * x - dt * self.apply(x))))
            if record:
                residuals.append(actual)
            if actual <= threshold:
                return x, SolveReport(True, i, tuple(residuals))
            z = r if precondition is None else precondition(r)
            updated = torch.sum(r * z)
            p, rz = z + (updated / rz) * p, updated
        return x, SolveReport(False, self.maxiter, tuple(residuals))

    def solve(self, rhs, *, shift, dt, out, diagnostics):
        if out is not None and out is not rhs:
            raise ValueError("implicit solve output must alias its RHS")
        e = self.engine
        value = e.transform_backend.inverse(rhs, e.boundaries)
        solution, report = self.iterate(
            value, shift=shift, dt=dt, record=diagnostics is not None,
            precondition=self.precondition if self.preconditioner is not None else None,
        )
        if diagnostics is not None:
            diagnostics["linear_solve"] = report
        if not report.converged:
            raise RuntimeError("test diffusion CG did not converge")
        result = e.transform_backend.forward(solution, e.boundaries)
        return result if out is None else out.copy_(result)


def variable_engine(dt, shape, forcing, *, maxiter=128, precondition=None):
    def explicit(_u, time):
        return forcing(time)

    return reference_engine(
        assemble_periodic(_ScalarModel(1, explicit), dt, shape), torch.float64, False,
        tier="matrix_free_test",
        factory=lambda e: VariableDiffusion(e, maxiter=maxiter, precondition=precondition),
    )
