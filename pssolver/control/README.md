# PSSolver discrete-adjoint control

This package provides a correctness-first direct-adjoint loop for a built
`SpectralSolver`. It keeps the production in-place integrator unchanged and
uses a functional one-step adapter during optimization.

## Control problem

The initial implementation optimizes bounded, piecewise-constant amplitudes on
fixed smooth spatial masks:

```text
alpha(x, t) = baseline outside the masks
            + bounded amplitude(block(t), mask) inside each mask
```

This is intentionally lower-dimensional than a free `(t, x, y, z)` activity
field. It is suitable for validating the adjoint before scaling the control
space.

## Channel integration

The ready-to-run driver is `Channel_dal.py`. Its default continuation problem
uses the saved pure-splay initial loop and the nearby generated target:

```text
initial: Q_initial_pure_loop.npy  (loop center x = 64)
target:  data_pure_splay_loop_x62/Q_pure_splay_loop.npy  (loop center x = 62)
```

## Loop convention

Loop labels follow Yingyou Ma's *3D Dry Uniaxial Active Nematics in Bulk*.
The principal axes are the decreasing-eigenvalue eigenvectors of the
box-averaged orientational second moment

```text
q_box = <n n>_box,   lambda_N >= lambda_M >= lambda_L.
```

The principal plane is N-M and its normal is L. The loop's geometric plane
normal is the separate vector `nu`, while the director-rotation axis is
`Omega`. Under the ideal-initial-loop assumptions, `k || nu` and `Omega || L`.
Classification refers to the complete pi-rotation 1D skyrmion crossing the
loop:

```text
k || N  -> pure splay
k || M  -> pure bend
k || L  -> pure twist
```

This is not the small-perturbation mapping, where `k || N` is bend and
`k || M` is splay. Plaquette orientation is reported separately as
`yz_plane_topology`; it describes loop-line geometry and cannot classify the
surrounding texture as splay, bend, or twist.

Both tensors have shape `512 x 40 x 40 x 5`. The default stride `8 2 2` reduces
them to `64 x 20 x 20`; the stricter geometry runs use `--stride 4 2 2`, or
`128 x 20 x 20`. Restriction is conservative block averaging, and the default
`--target-mode translated_initial` constructs the coarse target by an exact
periodic Fourier translation of the restricted initial field. This removes
sampling-phase differences between separately downsampled initial and target
files. Unless `--mask` is supplied, the
driver uses the significant initial-target difference to define the x path.
When loop metadata is available, it multiplies that path by a localized y-z
envelope around the line joining the initial and target loop centers. The
envelope extends beyond the loop radius by `--transverse-mask-margin`, uses
`--transverse-mask-transition-width` for smooth decay, and tapers near the
nonperiodic walls according to `--control-wall-buffer`. Inputs without loop
metadata retain the smooth difference-box fallback.

Two spatial bases are available through `--mask-type`:

- `path_gaussian` (default) partitions the path envelope into `--num-masks`
  normalized Gaussian windows along periodic x.
- `rbf_3d` places a normalized tensor grid of three-dimensional Gaussian RBFs
  inside the same envelope. `--rbf-counts NX NY NZ` sets the center counts;
  the default `8 3 3` produces 72 masks.

With 20 steps and `--block-size 5`, the default path basis optimizes a
`4 x 8` matrix of bounded activity amplitudes:

```text
alpha(x,t) = sum_i amplitude[time_block(t), i] * mask_i(x)
```

The partition sums exactly to the original path mask, so overlap does not
increase the activity bound. `--num-masks`, `--mask-overlap`, `--block-size`,
`--temporal-weight`, and `--spatial-weight` configure the control basis and its
time/space smoothness. The optimized matrix and basis are saved as
`alpha_amplitudes.csv`, `alpha_amplitudes.npy`, and `control_masks.npy`.
For `rbf_3d`, spatial regularization is evaluated between neighboring centers
along all three RBF-grid axes.

```bash
# Current localized x-path basis
conda run -n Nematics3D python Channel_dal.py --mask-type path_gaussian

# Full 3D RBF basis with 8 x 3 x 3 centers
conda run -n Nematics3D python Channel_dal.py \
  --mask-type rbf_3d --rbf-counts 8 3 3
```

The tracking objective is core aware. In addition to target-core-weighted Q
error, it compares a differentiable low-order-parameter core density and
penalizes loss of total core mass. This prevents the optimizer from reducing
the tracking loss by simply annihilating the loop. A uniform no-loop reference
and the initial, target, and optimized loss components are saved to
`loss_components.json`.

The objective is selected through a small registry using `--loss-function`.
The legacy default remains `core_aware_q_tracking`. For translation without a
prescribed intermediate path, use `free_path_shape_preserving`:

```bash
conda run -n Nematics3D python Channel_dal.py \
  --loss-function free_path_shape_preserving
```

This objective constrains only the terminal core center to the target. It does
not prescribe a center trajectory `C_ref(t)`. During the trajectory and at the
terminal state it penalizes changes in full-domain soft-core mass, the
centered three-dimensional second moment, and the Q texture after differentiable
comoving alignment. The state geometry is deliberately measured on the full
domain rather than through the actuator mask, so changing the control support
cannot change the definition of loop position or shape.

Registered losses must provide `stage_cost(q, alpha, step)`,
`terminal_cost(q)`, and `components(q, alpha=None)`. New objectives can be
added to `LOSS_FUNCTIONS` in `pssolver/control/objectives.py` without changing
the DAL forward/adjoint loop.

```bash
conda run -n Nematics3D python Channel_dal.py
```

The first physical-horizon continuation run uses ten temporal blocks and
eighty control variables:

```bash
conda run -n Nematics3D python Channel_dal.py \
  --steps 1000 \
  --block-size 100 \
  --checkpoint-stride 10 \
  --iterations 10 \
  --skip-gradient-check \
  --output data_dal_pure_splay_x64_to_x62_T1
```

Optimization uses a fixed pressure-CG iteration count so perturbed controls
execute the same differentiable forward map. Check the pressure residuals
separately when changing grid size, timestep, or activity bounds.

Before increasing the horizon or grid size, compare the production integrator
against the DAL functional step on the same pure-splay initial state and
time-varying activity:

```bash
conda run -n Nematics3D python scripts_channel/validate_dal_forward.py
```

The validation writes per-step Q, velocity, and pressure errors to
`data_dal_forward_validation/forward_consistency.csv`, a pass/fail summary to
`summary.json`, and both final Q tensors for direct inspection. Pressure warm
starts are cleared in both paths so the comparison tests the discrete forward
map rather than differences in iterative-solver history.

After forward and gradient consistency pass, calibrate the physical horizon
using several constant activity branches:

```bash
conda run -n Nematics3D python scripts_channel/calibrate_pure_splay_forward.py
```

The default run advances `alpha = 0, 2.5, 5.0` for 1000 steps on the
`64 x 20 x 20` grid. It records the low-order-parameter core center and radius,
geometric plaquette winding counts, periodic template displacement, velocity,
and normalized target mismatch. Branch summaries distinguish stationary
motion, motion toward or away from the target, and topology/numerical failure.

After constructing and building the same solver used by the uncontrolled
channel simulation:

```python
from pssolver.control import (
    DiscreteAdjointLoop,
    FunctionalSemiImplicitStep,
    QTrackingObjective,
    TemporalMaskControl,
    load_q_target,
    partition_mask_along_axis,
    smooth_box_mask,
)

initial_q = solver.fields.spatial[:5].detach().clone()
target_q = load_q_target("data_target/Q_800.npy", device=device)

mask = smooth_box_mask(
    (Nx, Ny, Nz),
    bounds=((246, 266), (10, 30), (10, 30)),
    transition_width=2.0,
    device=device,
)
masks = partition_mask_along_axis(
    mask,
    8,
    axis=0,
    bounds=(246, 266),
    overlap=0.75,
)
control = TemporalMaskControl(
    masks,
    num_steps=control_steps,
    block_size=20,
    batchsize=batchsize,
    alpha_min=0.0,
    alpha_max=5.0,
    initial_alpha=2.5,
)
objective = QTrackingObjective(
    target_q,
    dt=dt,
    spatial_mask=mask,
    running_weight=0.1,
    terminal_weight=1.0,
    control_weight=1e-4,
)
dal = DiscreteAdjointLoop(
    FunctionalSemiImplicitStep(solver),
    objective,
    control,
    num_steps=control_steps,
    checkpoint_stride=20,
    temporal_control_weight=1e-3,
    spatial_control_weight=1e-3,
)
```

``load_q_target`` accepts only canonical active-nematic Q data. The target
``Q_*.npy`` and its schema-v1 ``metadata.json`` must be in the same
directory; legacy or unknown conventions are rejected.

Always run directional derivative checks on a reduced grid and short horizon
before optimization. The driver checks the gradient direction and multiple
seeded random directions at several finite-difference step sizes, then saves
the complete table to `gradient_check.json`:

```python
print(dal.directional_derivative_check(initial_q, epsilon=1e-3))
print(dal.taylor_test(initial_q))
```

Then perform DAL iterations. `Channel_dal.py` supports normalized steepest
descent and limited-memory BFGS directions with Armijo backtracking. It also
stops only after both relative cost change and gradient criteria remain below
their tolerances for `--convergence-patience` accepted iterations:

```python
for iteration in range(20):
    result = dal.armijo_step(
        initial_q,
        initial_step=1.0,
        direction_method="lbfgs",
        lbfgs_history_size=10,
    )
    print(iteration, result)
    if not result.accepted:
        break
```

## Active force requirement

For spatially varying activity the momentum forcing must be computed as
`div(beta * alpha * Q)`, not `beta * alpha * div(Q)`. Use
`active_force_divergence()` from this package when defining the channel static
model.

## Scaling limitation

The reference implementation differentiates the operations executed by the
static solver during each one-step VJP. That is appropriate for small-grid
gradient validation. Before full `512 x 40 x 40` optimization, replace the
unrolled pressure-CG backward with an implicit custom adjoint solve and verify
its transpose using a dot-product test.
