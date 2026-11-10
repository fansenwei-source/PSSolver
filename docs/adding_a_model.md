# Adding an internal engine-v2 model

The component API is internal until S-28; the released public API is unchanged.

1. Write the PDE and units. Here the dimensionless periodic Allen–Cahn equation is
   `∂φ/∂t = ε²∇²φ + φ − φ³`.
2. Declare fields and parameters in class order.
3. Declare a uniform diagonal linear expression and a physical explicit kernel:

```python
class AllenCahn(Model):
    """∂φ/∂t = ε² ∇²φ + φ − φ³ on a dimensionless periodic box."""

    phi = EvolvedField()
    epsilon = Parameter(default=0.05)

    def terms(self, f, p):
        yield Linear(f.phi, p.epsilon**2 * lap(f.phi))
        yield Explicit(f.phi, lambda phi: phi - phi**3)
```

4. Build `AssemblyProblem` with explicit geometry, boundary bases, Euler time
   step, spectral numerics, execution and observables. Call `assemble`, then
   `assembled.engine(device="cpu", dtype=...)`. See `examples/allen_cahn.py`.
5. Run the checks in `examples/model_template/checks.py`: SymPy-derived forcing,
   spatial and temporal convergence, invariant/energy evolution, exact restart
   and Taylor gradients. Check pure/in-place and independent batch members too.

Physical kernel inputs and outputs are `(components, batch, *shape)`; outputs
must have the exact declared shape. Coordinate inputs are `(3, 1, *shape)` and
time inputs are Python floats. No implicit output broadcast is accepted.
Unsupported geometry, integration, boundary and solver capabilities carry
explicit scope or mathematical rejection codes; they do not silently fall back.
