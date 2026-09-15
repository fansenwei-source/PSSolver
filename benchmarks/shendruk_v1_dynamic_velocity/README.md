# Shendruk V1 benchmark with dynamic velocity fields

This directory contains a self-contained benchmark driver in which all five
independent Q components and all three velocity components are registered as
PSSolver dynamic fields. Pressure remains a Lagrange multiplier rather than an
independent dynamic field.

The evolved equations are

\[
(\partial_t+\mathbf u\cdot\nabla)Q-S(W,Q)=H/\gamma,
\]

\[
\rho(\partial_t\mathbf u+\mathbf u\cdot\nabla\mathbf u)
=-\nabla p+\eta\nabla^2\mathbf u-\mathrm{fric}\,\mathbf u
+\nabla\cdot\Pi_{\mathrm{nem}},
\qquad \nabla\cdot\mathbf u=0.
\]

The velocity fields cannot use the unmodified component-by-component
`SemiImplicitEulerIntegrator`: pressure couples the three components through
incompressibility. `IncompressibleNavierStokesEulerIntegrator` therefore
updates Q normally, then advances the three velocity spectra together with a
mixed FFT/DCT/DST Schur solve.

The default benchmark choices are:

- V1 analytic periodic defect-gas initialization with a small coherent
  wall-normal twist;
- zero initial velocity;
- density `rho=1`;
- no physical friction;
- dynamically evolved uniform tangential momentum;
- `cubic_half` dealiasing;
- periodic x/y, Neumann Q and tangential velocity in z, and Dirichlet normal
  velocity in z.

Example dry run:

```bash
python benchmarks/shendruk_v1_dynamic_velocity/Plane_v1_dynamic_velocity.py \
  --activity-number 18 \
  --height 20 \
  --initialization-protocol v1 \
  --output-dir /tmp/shendruk_v1_dynamic_velocity_dry \
  --device cpu \
  --dtype float64 \
  --disable-spectral-refresh \
  --dry-run
```

This is a runnable research prototype, not yet an accepted production
benchmark. Before a long run it still requires a one-step GPU preflight and
time-step/spatial convergence checks for the inertial formulation.

## Matched Fig. 4 workflow

`fig4_workflow.py` freezes the exact 23-point matrix used by the archived
2026-09-04 V1 preview:

- `H=10`, `Nz=40`, `A=5,10,15,17,18,20,22`;
- `H=15`, `Nz=60`, `A=7,12,16,17,18,20,25,33`;
- `H=20`, `Nz=80`, `A=9,15,17,18,20,25,35,44.7`.

Every run uses `Nx=Ny=320`, `Lx=Ly=100`, `dt=0.005`, `T=200`, seed 24,
float64 arithmetic, disabled TF32 and spectral refresh, `cubic_half`
dealiasing, and the archived V1 initial condition. Q snapshots are retained
at `T=100,105,...,200`. Formal runs remain Q-only; the one-step preflight also
saves u and p so the inertial saddle solve can be checked.

Create a plan bound to the current commit and source hashes:

```bash
python benchmarks/shendruk_v1_dynamic_velocity/fig4_workflow.py plan \
  --output /path/to/control/fig4_plan.json
```

Print the preflight command without executing it:

```bash
python benchmarks/shendruk_v1_dynamic_velocity/fig4_workflow.py command \
  --plan /path/to/control/fig4_plan.json \
  --output-root /path/to/output \
  --preflight
```

Execution additionally requires both `--execute` and
`--confirm-direct-execution`, a clean worktree, a matching commit, unchanged
source hashes, and a new run directory. Validate the completed preflight with:

```bash
python benchmarks/shendruk_v1_dynamic_velocity/fig4_workflow.py validate \
  --plan /path/to/control/fig4_plan.json \
  --output-root /path/to/output \
  --preflight \
  --report /path/to/control/preflight_validation.json
```

This workflow reproduces the archived preliminary single-seed scan design.
It does not upgrade that design into a strict or statistically converged
reproduction of the published paper. The intentional physics differences
from the archived quasistatic run are inertial dynamic velocity and an evolved
uniform tangential momentum mode (`rho=1`, `fric=0`, `mean_flow_policy=evolve`).
