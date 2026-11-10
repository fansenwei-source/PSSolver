# Internal model template

1. State the equation, units, boundaries and invariants.
2. Declare fields and fixed/control parameters in `model.py`.
3. Split uniform diagonal expressions and physical explicit kernels.
4. Compose an `AssemblyProblem`; unsupported capabilities fail before tensor work.
5. Use `checks.py` for manufactured solutions, convergence, energy, restart and
   Taylor checks. Record all samples and use the model's preregistered gates.

This is an internal S-26 interface, not a new top-level public API. See
`examples/allen_cahn.py` and `tests/test_v0_5_s26_allen_cahn.py` for a complete use.
Neither this template nor examples are included in the wheel or sdist.
