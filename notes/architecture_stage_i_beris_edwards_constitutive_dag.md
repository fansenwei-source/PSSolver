# Architecture Stage I: Beris--Edwards constitutive DAG

## Scope and result

Stage I migrates the complete one-constant Beris--Edwards nematic force into
the opt-in algebraic dependency architecture.  The new path expresses

```text
Q -> H -------------------\
 \-> grad(Q) -> stress ----> force -> geometry-specific Stokes -> u,p
```

as five separately owned algebraic systems.  It reuses the equation helpers
and Plane stress-divergence operators already qualified by the issue 6 and
issue 7 work; it does not rederive or fork a second constitutive convention.

This stage deliberately keeps Q static.  The zero explicit RHS is a
qualification harness for constitutive assembly, field lifetimes, boundary
projection, and Stokes consumption.  Advection, flow alignment, molecular
relaxation in the Q evolution, production initialization, driver migration,
and trajectory equivalence belong to the next coupled-model stage.

No production entry point, numerical default, benchmark script, Plane solver,
or Channel solver selects this path.

## I1: physical parameters and canonical tensor layout

`BerisEdwardsConstitutiveParameters` owns only physical coefficients:

```text
A, B, C, L1
flow-alignment parameter
active-stress prefactor
```

The compact symmetric-traceless ordering remains

```text
(Qxx, Qxy, Qxz, Qyy, Qyz),  Qzz = -Qxx - Qyy.
```

The molecular field uses the same compact ordering.  Both stress parts use
the complete row-major nine-component ordering

```text
(xx, xy, xz, yx, yy, yz, zx, zy, zz).
```

Every coefficient is finite and `L1` is positive.  Numerical choices such as
physical versus spectral molecular-field assembly, pointwise execution, and
physical versus spectral divergence summation are not physical parameters.
They live in `PlaneBerisEdwardsSolverOptions` and are reported by the resolved
executor metadata.

## I2: molecular field and stress equations

The raw thermodynamic molecular field is

```text
H = -A Q
    -B [Q^2 - I tr(Q^2)/3]
    -C tr(Q^2) Q
    +L1 Laplacian(Q).
```

It is not divided by the rotational viscosity.  That distinction is required
because the stress law consumes raw H.

With `M = Q + I/3`, the algebraic reactive plus active stress is

```text
Pi_alg = 2 lambda M (Q:H)
         -lambda (H M + M H)
         +Q H - H Q
         +active_prefactor Q.
```

The one-constant distortion stress is

```text
Pi_dist_ij = -L1 partial_i Q_kl partial_j Q_kl.
```

The physical formulas are still implemented by the reusable functions in
`pssolver.models.active_nematics.beris_edwards`.  Stage I executors select
projection and evaluation order around those functions; they do not contain a
different physical formula.

## I3: field lifetimes and ownership

The Stage I qualification model declares:

- five evolved Q components;
- five transient H components;
- fifteen transient Q-gradient components;
- nine transient algebraic-stress components;
- nine transient distortion-stress components;
- three transient projected force components;
- three stored algebraic velocity components and one stored algebraic
  pressure component.

Thus 41 constitutive intermediates are present in the spectral plan and the
pre-RHS cache, but absent from legacy persistent field storage and restart
tensors.  The legacy state contains only nine components: five Q components,
three velocity components, and pressure.

Systems are deliberately declared in reverse dependency order.  The Stage H
planner freezes the actual execution order as

```text
molecular_field
q_gradient
nematic_stress
nematic_force
flow
```

The component-labelled graph therefore makes every input to stress and force
auditable.  Stokes consumes only the final force components; it does not own
or recompute Q, H, or nematic stress.

## I4: Plane parity and complete-force projection

For the current free-Q Plane model, Q, H, tangential velocity, and pressure
use periodic/periodic/Neumann space.  Wall-normal Q derivatives use
periodic/periodic/Dirichlet space.  The distortion stress is wall-odd exactly
for `xz`, `yz`, `zx`, and `zy`; its other five components are wall-even.

The algebraic stress is represented in Q's common wall-even space.  Its
row-wise divergence contains tangential and wall-normal derivatives with
different parity.  Consequently, the complete physical force must be formed
with the qualified mixed-basis Plane rules and then projected into

```text
force_x, force_y : periodic / periodic / Neumann
force_z          : periodic / periodic / Dirichlet.
```

Stage I therefore does not replace the existing Plane force construction with
three generic `divergence()` calls.  It explicitly dispatches the capability

```text
(PlaneSlab, beris_edwards_force)
    -> plane_projected_complete_stress_divergence
```

and reuses `projected_common_basis_stress_divergence()` plus
`projected_distortion_stress_divergence()`.  This preserves the previously
validated DCT/DST parity split and makes the final velocity-space projection a
named numerical operation rather than an accidental side effect.

The remaining systems also use exact Plane registrations.  No PeriodicBox or
RectangularChannel fallback is provided for the constitutive capabilities.
The independently qualified Stage G Stokes registry is then composed into the
same construction registry.

## I5: mean active force remains a modeling choice

Free Q anchoring does not imply zero mean active tangential force.  For
example, a Neumann `Qxz` profile can produce a nonzero discrete mean of
`partial_z Qxz`.  The Stage I force retains this mean before the Stokes
zero-mode policy is applied.

The qualification test constructs exactly this case and observes both:

- a nonzero mean `force_x`;
- zero mean `ux` under the explicit `zero_mean` tangential policy.

This confirms that deleting the uniform tangential flow is a model/reference-
frame choice, not a pressure gauge and not an automatic consequence of free Q
anchoring.  The alternative positive-friction policy remains represented by
the existing typed Stokes request.

## I6: equation and compatibility qualification

Stage I tests validate the new path at each dependency boundary:

1. physical- and spectral-linear molecular-field assembly against the complete
   molecular-field helper;
2. all fifteen native projected Q derivatives against the mathematical
   operator context;
3. all eighteen stress components against the reusable constitutive helpers;
4. complete row-wise stress divergence against the qualified Plane parity-
   split operators in physical and spectral summation modes;
5. the projected force and all four Stokes spectral outputs against an
   independently constructed `BerisEdwardsFreeSlipStokes` production oracle;
6. full-complex/full-transform and the qualified Hermitian-half/truncated
   transform paths;
7. lifecycle counts, deterministic DAG order, metadata separation, and
   continued production isolation.

The existing full-matrix, automatic-differentiation, energy-exchange, wall-
power, manufactured-force, Schur-pressure, and free-slip Stokes tests remain
part of the focused qualification set.  Stage I adds architectural coverage;
it does not weaken their scientific tolerances.

Desktop CPU qualification after the final implementation:

- Stage F--I architecture plus existing Beris--Edwards force, energy, stress,
  and free-slip Stokes suites: `87 passed`;
- complete repository suite: `749 passed, 8 subtests passed`.

Both commands use an external Python bytecode cache and disable the pytest
cache.  No GPU qualification is required because Stage I neither changes a
production path nor promotes a numerical default.

## Production boundary after Stage I

Stage I proves that the new architecture can assemble the complete current
Plane constitutive force and feed it to the already qualified geometry-specific
Stokes solver.  It does not yet claim a migrated Beris--Edwards simulation.

The next stage should add the complete Q explicit RHS on top of this frozen
dependency chain, including B/C bulk relaxation, material advection, full flow
alignment, and co-rotation, while keeping the A/L1 part in the diagonal IMEX
operator.  That opt-in coupled Plane model must then pass single-step and
multi-step trajectory comparisons against the production benchmark path before
driver migration or GPU performance qualification is considered.

Channel constitutive parity, Channel-specific active-nematic boundary choices,
production checkpoint coordination, and any default promotion remain separate
later decisions.
