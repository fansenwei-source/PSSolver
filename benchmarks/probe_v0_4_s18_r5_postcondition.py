"""V-61 symmetric path-level fault probe, adapted from R-0057.

Runs only on CPU. Invoke with <repo> <production|transpose|implicit> <n>.
Each invocation uses a fresh process; source trees are never modified.
"""

if __name__ == "__main__":
    import json, sys, types
    repo, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.path.insert(0, repo)
    import warnings; warnings.simplefilter("ignore", DeprecationWarning)
    import torch
    from pssolver import SpectralSolver
    from pssolver.functional import ChannelPressureTransposeOperator, ChannelImplicitPressureAdjoint
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver
    import pssolver.linear_solvers.stokes.channel_no_slip as nsp
    import pssolver.functional.channel_pressure as cp
    import pssolver.functional.channel_pressure_adjoint as cpa
    shape, dtype = (12, 8, 32), torch.float32
    spectral = SpectralSolver(shape, L=tuple(8.0 * (1.0 + 0.37 * v) for v in shape), device="cpu", dtype=dtype)
    mod = {"production": nsp, "transpose": cp, "implicit": cpa}[path]
    st = {}
    real_sat = mod.channel_pressure_fixed_work_roundoff_saturated
    def sat(rel, dt):
        st["calls"] += 1
        r = real_sat(rel, dt)
        if st["inject_at"] is not None and st["calls"] == st["inject_at"]:
            st.update(force_unequal=True, forced={"real_represented": r, "rel": float(rel)})
            return False
        return r
    mod.channel_pressure_fixed_work_roundoff_saturated = sat
    class TorchProxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)
    proxy = TorchProxy("torch")
    def fake_equal(a, b):
        if st.get("force_unequal"):
            st.update(force_unequal=False, nan_next=True, equal_forced={"real_equal": bool(torch.equal(a, b))})
            return False
        return torch.equal(a, b)
    proxy.equal = fake_equal
    mod.torch = proxy
    def run(inject_at):
        st.clear(); st.update(calls=0, inject_at=inject_at, nan_next=False, nan_injected=False, forced=None, equal_forced=None)
        solver = ChannelNoSlipModalStokesSolver(spectral.transform_backend, friction=1e-3, viscosity=0.73,
                                                pressure_relative_tolerance=1e-6, pressure_max_iterations=n, pressure_fixed_iterations=n)
        tr = ChannelPressureTransposeOperator(solver)
        g = torch.Generator().manual_seed(0)
        rhs = tr._project_gauge(torch.complex(torch.randn((1, *shape), generator=g, dtype=dtype), torch.randn((1, *shape), generator=g, dtype=dtype)))
        if path == "production":
            owner, attr = solver, "_pressure_operator"
        elif path == "transpose":
            owner, attr = tr, "apply_pressure_operator_transpose"
        else:
            implicit = ChannelImplicitPressureAdjoint(solver)
            owner, attr = implicit._operators, "apply_pressure_operator"
        real_op = getattr(owner, attr)
        def op(x):
            y = real_op(x)
            if st["nan_next"]:
                st.update(nan_next=False, nan_injected=True)
                return torch.full_like(y, float("nan"))
            return y
        owner.__dict__[attr] = op
        try:
            if path == "production":
                res = solver._solve_pressure(rhs.clone())
                info = dict(termination=solver.last_pressure_termination_reason, acceptable=solver.last_pressure_acceptable, iterations=solver.last_pressure_iterations)
            elif path == "transpose":
                res = tr.solve_pressure_transpose(rhs.clone())
                d = tr.last_transpose_diagnostics
                info = dict(termination=d.termination_reason, iterations=d.iterations)
            else:
                res = implicit.solve(rhs.clone())
                d = implicit.last_primal_diagnostics
                info = dict(termination=d.termination_reason, iterations=d.iterations)
            out = dict(outcome="returned", finite=bool(torch.isfinite(res).all()), **info)
        except Exception as exc:
            det = getattr(exc, "details", None)
            out = dict(outcome="raised", exc=type(exc).__name__, message=str(exc)[:160],
                       termination=det.get("termination_reason") if isinstance(det, dict) else None)
        out.update(saturated_calls=st["calls"], forced=st["forced"], equal_forced=st["equal_forced"], nan_injected=st["nan_injected"])
        return out
    rec = run(None)
    inj = run(rec["saturated_calls"]) if rec.get("termination") == "fixed_work_completed" else None
    print(json.dumps({"path": path, "n_fixed": n, "record": rec, "inject": inj}, sort_keys=True))
