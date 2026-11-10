"""Trusted SymPy manufactured source for S-26's spatial convergence contract."""

import sympy as sp
import torch

from examples.allen_cahn import AllenCahn
from examples.model_template.checks import manufactured_source
from pssolver.core.terms import Coordinates, Explicit, Linear, Time, lap


def expressions():
    x, y, z, t = sp.symbols("x y z t", real=True)
    epsilon = sp.Rational(1, 20)
    a = sp.Rational(3, 10) * sp.exp(sp.sin(2*sp.pi*x) + sp.cos(2*sp.pi*y)) * sp.cos(2*sp.pi*z)
    exact = a + t / 10
    source, provenance = manufactured_source(exact, coordinates=(x,y,z), time=t,
        diffusion=epsilon**2, reaction=lambda phi: phi - phi**3)
    modules = [{"sin": torch.sin, "cos": torch.cos, "exp": torch.exp}]
    return (sp.lambdify((x,y,z,t), exact, modules=modules),
            sp.lambdify((x,y,z,t), source, modules=modules), provenance)


class ForcedAllenCahn(AllenCahn):
    def __init__(self):
        super().__init__()
        self.exact, self.source, self.provenance = expressions()

    def terms(self, f, p):
        yield Linear(f.phi, p.epsilon**2 * lap(f.phi))
        yield Explicit(f.phi, lambda phi, xyz, time: phi - phi**3 + self.source(*xyz, time).unsqueeze(0),
            inputs=(f.phi, Coordinates(), Time()))
