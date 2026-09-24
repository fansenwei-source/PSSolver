# Phase 7 P7.7.3: capability resolution and tensor-free lowering

Status: `P7_7_3_COMPLETE_P7_7_4_NOT_AUTHORIZED`.

Parent baseline: `2186191d22a007ed39c9655e06ee10f1dfdd8d37` on
`next/pssolver-v0.2.0-architecture`.

Classification:
`PASS_P7_7_3_CAPABILITY_RESOLUTION_TENSOR_FREE_LOWERING`.

P7.7.3 converts the composed `SimulationSpec` from P7.7.2 into a frozen,
deterministic `SimulationLoweringPlan`. The plan describes the exact spectral
spaces, derivative parity transitions, capability implementations, nullspace
semantics, and geometry-specific Stokes solver that a later execution builder
would need. It does not import or construct that execution machinery.

The machine-readable authority is
[phase_7_p773_capability_lowering.json](phase_7_p773_capability_lowering.json).

## Strict support boundary

The resolver is intentionally a closed whitelist. It admits exactly:

1. `complete_stress_beris_edwards` on `plane_slab` with the qualified
   periodic/periodic/free-slip wall spaces;
2. `legacy_active_force_active_nematics` on `rectangular_channel` with the
   qualified periodic/no-slip/no-slip velocity spaces.

It does **not** claim that complete-stress Beris--Edwards is qualified in the
Channel, that the legacy active-force model is qualified in Plane, or that
arbitrary model and geometry declarations can be exchanged. There is no
nearest-match or legacy fallback.

`SimulationLoweringError` owns a `LoweringRejection` containing a stable
`LoweringRejectionCode`, human-readable message, and canonical JSON context.
Unsupported combinations therefore fail before tensor allocation with an
auditable reason. Rejections cover model/geometry, dimension/topology,
capability set, integrator, backend, Hermitian axis, asymmetric faces,
boundary signature or semantic, equation layout, Stokes contract, derived
space, nullspace policy, and solver options.

## Physical boundaries remain distinct from modal parity

`AxisBasisRequirement.modal_kind` labels periodic/even/odd modal parity. It is
not automatically a prescribed physical boundary condition. Each component
space records one of five provenance values:

- `physical_boundary`;
- `algebraic_compatibility`;
- `derivative_parity`;
- `constitutive_parity`;
- `projected_output`.

Thus Q and velocity spaces descend from physical face laws, pressure descends
from algebraic compatibility, gradient and distortion spaces are derived,
and force components name the spaces into which they are projected for the
Stokes solve. This prevents a DCT/DST implementation choice from being
mistaken for new wall physics.

The P7.7.2 face representation can express different lower and upper wall
laws, but the current eigenbasis implementation cannot lower such a pair.
P7.7.3 therefore rejects asymmetric pairs explicitly. It does not silently
replace them with a symmetric DCT or DST space. Lifting, tau methods, strong
anchoring, and nonhomogeneous data remain future capabilities.

## Exact derivative rules

Every Q and velocity gradient is lowered separately in canonical axis-major
order. The modal maps are frozen as:

```text
FFT --d/dx--> FFT : Fourier i k
DCT --d/dx--> DST : negative sine multiplier
DST --d/dx--> DCT : positive cosine multiplier
```

Only the differentiated axis changes basis. Each `DerivativeRequirement`
records the capability, source component, output component, axis, multiplier,
and complete input/output basis signatures.

For Plane this reproduces Q and tangential velocity in FFT/FFT/DCT, normal
velocity in FFT/FFT/DST, wall-normal Q gradients in the odd space, and
wall-normal derivatives of normal velocity in the even space.

For Channel it reproduces Q and pressure in FFT/DCT/DCT and all velocity
components in FFT/DST/DST, with each bounded derivative toggling exactly one
DCT/DST parity.

## Constitutive and force spaces

For the complete Plane model:

- molecular field and algebraic stress use the wall-even Q space;
- distortion components `xz`, `yz`, `zx`, and `zy` use the wall-odd space;
- the other five distortion components use the wall-even space;
- the compatibility `distortion_odd_z` declaration is checked against the
  independently derived odd space;
- complete stress divergence is bound to
  `plane_projected_complete_stress_divergence` and projected into the three
  velocity spaces.

For the legacy Channel model, active-force derivatives can have different
bounded-axis parities before they are summed in physical space. The lowering
therefore records the qualified strategy as
`physical_mixed_parity_sum_then_velocity_space_projection`; it does not claim
that the unprojected physical sum has one native spectral basis.

## Capability bindings

Each declared equation capability is bound exactly once to a tensor-free
implementation identifier and strategy. Plane resolves seven capabilities:

- complete Beris--Edwards Q evolution;
- Q gradient;
- velocity gradient;
- one-constant molecular field;
- complete one-constant nematic stress;
- projected complete-stress divergence;
- free-slip modal incompressible Stokes.

Channel resolves five capabilities:

- legacy active-force Q evolution;
- Q gradient;
- velocity gradient;
- legacy active-force divergence;
- no-slip Channel incompressible Stokes with pressure PCG.

The identifiers are declarations, not imported callables. Runtime registries
or string dispatch do not enter a timestep.

## Geometry-specific solvers and nullspaces

Plane requires:

```text
pssolver.linear_solvers.stokes.plane_free_slip.FreeSlipModalStokesSolver
```

The plan preserves zero-mean pressure and distinguishes the two qualified
tangential-mode choices:

- `zero_mean`, zero friction: remove the uniform tangential velocity and force
  modes as an explicit modeling choice;
- `friction`, positive friction: retain the modes and resolve them with drag.

This is not treated as a pressure gauge.

Channel requires:

```text
pssolver.linear_solvers.stokes.channel_no_slip.ChannelNoSlipModalStokesSolver
```

The plan preserves the pressure-PCG controls and zero-mean pressure gauge.
Uniform velocity modes are `not_applicable` because two bounded no-slip axes
remove that nullspace.

No concrete solver module is imported while creating either plan.

## Identity and dependency boundary

`SimulationLoweringPlan` stores the complete source `SimulationSpec` SHA-256
and has its own canonical SHA-256. Changing grid shape changes the plan; two
equal inputs produce equal frozen plans and identical hashes.

The generic plan products live in `pssolver.planning.simulation`, which
depends only on `core` and `planning`. The composition resolver lives in
`pssolver.configuration.simulation_lowering`. Its two exact planning imports
and one Stokes declaration import are recorded in the dependency ratchet.

Neither provisional module is exported at a package root. Existing
applications, runtimes, public facades, pickle identities, checkpoints,
metadata schemas, and defaults remain unchanged.

## Verification

Targeted P7.7.0--P7.7.3, core-contract, planning, and dependency tests:

```text
80 passed
```

Complete local CPU suite:

```text
2092 passed, 8 subtests passed
```

No GPU qualification is required because no runtime imports or consumes the
new plan, and the new modules cannot allocate a tensor or enter a timestep.

## Performance effect

There is no hot-path runtime or memory cost in P7.7.3. Existing simulations do
not call the resolver. A future connection must run it once before device
binding, convert the result to pre-bound objects, and keep capability lookup,
field-name lookup, and boundary dispatch outside the timestep.

## Next slice

P7.7.4 is **not authorized by this result**. A separately authorized slice may
connect these requirements to an opt-in construction path, prove object and
numerical identity against the qualified Plane and Channel implementations,
and keep both production defaults unchanged until independent CPU/H100
qualification passes.

P7.7.3 does not add boundary physics, perform the runtime connection, promote
a solver, authorize Phase 8, or authorize Phase 9.
