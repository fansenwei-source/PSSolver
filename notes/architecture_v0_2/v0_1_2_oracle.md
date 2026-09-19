# Frozen v0.1.2 numerical and performance oracle

This record freezes the reference facts used to judge v0.2 architecture work.
Performance values describe the qualified H100 configuration; they are not a
cross-hardware speed guarantee.

The machine-readable companion is `v0_1_2_oracle.json`.

## Source identity

- release tag: `v0.1.2`;
- annotated release commit:
  `4fa614616e9d67c98be52c150eaf303a0c80b5c1`;
- qualified bounded-dataflow candidate:
  `f121428d7a5e66e4b65c6e6f3777fdf1c8f8b23a`;
- H100 qualification Job: `10835210`;
- archived 103-entry manifest SHA-256:
  `65a72a34a42a526e8ada36bd436ff1cbbaed0d2213fb88804eed1db6248b7876`.

The source release passed 1,134 CPU tests and eight subtests.  Release wheel
and sdist validation, isolated installation, both command entry points, a
bounded dry run, and a one-step finite CPU smoke also passed.

## Scientific and numerical identity

The oracle is the supported three-dimensional Plane Beris--Edwards
active-nematic application coupled to quasistatic incompressible
Stokes--Brinkman flow.

The frozen compatibility boundary includes:

- the repository's single internal Q convention;
- free-slip velocity and Neumann Q Plane boundary assignments;
- transform basis, normalization, mode order, and component parity;
- dealias and spectral-storage semantics;
- pressure and tangential zero-mode conventions;
- float64 qualification with TF32 disabled;
- `legacy_production` as the supported default;
- production spectral refresh behavior;
- Q/u/p shape, dtype, and finite-value requirements;
- snapshot, diagnostics, metadata, checkpoint/restart, and final `COMPLETE`
  ordering.

## H100 performance reference

The candidate was measured against its frozen parent in one H100 allocation,
using three balanced trials per grid.

Execution environment:

- GPU: NVIDIA H100 PCIe;
- NVIDIA driver: `550.54.14`;
- CUDA toolkit/runtime: `12.1`;
- PyTorch: `2.5.1+cu121`;
- visible GPU count: `1`;
- TF32: disabled.

Profiler identity:

- lengths: `(100, 100, 20)`;
- dtype: float64;
- timestep: `0.005`;
- activity number: `18`;
- seed: `20260908`;
- dealias rule: `cubic_half`;
- projected transform execution: `truncated`;
- transform execution order: `real_first`;
- spectral storage: `hermitian_half`;
- molecular-field linear space: `spectral`;
- stress-divergence sum space: `spectral`;
- pointwise execution: `compile`;
- Q-gradient reuse: enabled;
- pressure diagnostics: disabled;
- spectral refresh during timing: disabled;
- timing scope: whole timestep;
- warm-up/profile steps: `10/50`;
- a fresh TorchInductor cache for each process;
- paired order: `A->B`, `B->A`, `A->B` at each grid.

| Grid | Parent mean ms/step | v0.1.2 mean ms/step | v0.1.2 median ms/step | Speedup |
|---|---:|---:|---:|---:|
| R128 (`128x128x32`) | 7.120957 | 5.750187 | 5.756708 | 1.238387x |
| R320 (`320x320x80`) | 48.960003 | 45.162243 | 45.142706 | 1.084091x |

Memory references:

| Grid | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| R128 | 0.396654 | 0.839844 |
| R320 | 5.721501 | 8.533203 |

Execution counters:

- forward transforms per step: `7`;
- inverse transforms per step: `32`;
- graph breaks: `0`;
- compile fallbacks: `0`;
- projected-transform fallbacks: `0`.

There was no OOM, NaN, Inf, CUDA error, or reserved-memory increase relative
to the parent.  The candidate was faster in all three paired trials at both
grids.

## Qualified 100-step output identity

The parent and candidate outputs were byte-for-byte identical:

| File | SHA-256 |
|---|---|
| `Q_100.npy` | `d20547158acba4d8ab6b9e3c6c0cc48954c6b4b6ce9af229fe826d510e577796` |
| `u_100.npy` | `1199675fe7c1a0dcc4ec1f6a0c9130021a30be94e33dab523325c22076ec5a64` |
| `p_100.npy` | `34ef4007db9f900f1902b5f3cd43efe83a82cbbc393dd8d0df6229613fc848ea` |

These hashes identify the archived qualification configuration.  A different
grid, parameter set, device, PyTorch build, or output schedule is not expected
to reproduce the same bytes.

The production trajectory used:

- activity number/height: `18/20`;
- box/grid: `100x100x20`, `128x128x32`;
- timestep/steps: `0.005/100`;
- seed: `24`;
- initialization: six-pair analytic periodic defect gas, extruded with the
  default Neumann-compatible twist (`0.01`, modes `1,2,3`);
- float64, TF32 disabled;
- `cubic_half`, `truncated`, `real_first`, `hermitian_half`;
- spectral molecular field and spectral stress-divergence summation;
- compiled pointwise kernels and Q-gradient reuse;
- `zero_mean`, friction `0`;
- hydrodynamic saving enabled;
- production-default spectral refresh enabled;
- step-100 Q, velocity, pressure, metadata, and `COMPLETE` required.

## Archived evidence location

The authoritative HPCC control directory recorded by the qualification is:

```text
/home/fansenwei/pssolver_bounded_dataflow_h100_ab_f121428_20260918_v1
```

Its principal files are `final_report.md`, `summary.json`, `summary.csv`, and
`checksums.sha256`.  The corresponding scratch root is:

```text
/scratch1/vincent/PSSolver/data_pssolver_bounded_dataflow_h100_ab_f121428_20260918_v1
```

The qualification instruction is preserved on the originating workstation at
`/home/fansenwei/Desktop/Develop/pssolver_v012_profile_a29bc1c_20260918_v1/`
with SHA-256
`7c499e182e6574f5233b3b98474e926d99f86991c560f84ff4b470e6b80ca6ea`.

The historical report did not publish a standalone canonical configuration
hash.  The explicit fields above plus the authoritative 103-entry manifest are
the frozen evidence available for v0.1.2.  Every future qualification must add
a canonical config hash and repository-relative manifest inventory rather
than repeating this gap.

## Architecture-oracle evidence

The separated canary remains a scientific oracle but is not the performance
oracle.  Its Stage O.4.1 R320 measurements relative to legacy were:

- timestep ratio: `1.429558205567479`;
- peak allocated ratio: `1.589530529807474`;
- peak reserved ratio: `1.328679331654841`.

Short and 100-step differences remained at roundoff scale, and same-backend
restart was byte-exact.  This combination is why v0.2 must preserve the clean
scientific contract while lowering to the qualified data path.

## Qualification rules for migration

### Mechanical extraction

- unchanged signatures and defaults;
- unchanged tensor layout and operation order;
- byte-identical short Q/u/p trajectory;
- complete CPU suite;
- installed import/facade check;
- H100 non-regression after the complete extraction, not after every move.

### Equivalent numerical reordering

- tolerances frozen before execution;
- raw and pressure-demeaned comparisons where applicable;
- shape, dtype, finite, boundary-residual, and transform-count checks;
- balanced H100 A/B with memory and fallback reporting.

### New operator, boundary, or model

- independent manufactured or analytic validation;
- convergence order and boundary residuals;
- conservation, dissipation, or free-energy checks where applicable;
- geometry-specific pressure/zero-mode tests;
- benchmark-scale validation before production promotion.

### Default promotion

Qualification of a candidate does not modify the default.  Promotion requires
a separate commit, implicit-default smoke, rollback selector, and explicit
review.

## Evidence references

- `notes/pssolver_v0_1_2.md`;
- `notes/pssolver_v0_1_scope.md`;
- `notes/architecture_stage_o_closure.md`;
- `notes/architecture_stage_o_production_migration_design.md`.
