# Periodic fast-path default promotion

The periodic fast path combines two independently selectable optimizations:

- contiguous transform groups use a zero-copy slice/view rather than advanced
  indexing;
- two or more periodic transform axes use one multidimensional
  `torch.fft.fftn`/`torch.fft.ifftn` call rather than a sequence of one-axis
  transforms.

The production defaults are:

```text
transform_group_indexing = contiguous_slice
periodic_transform_execution = multidim
```

The historical implementation remains an explicit rollback:

```text
transform_group_indexing = advanced
periodic_transform_execution = axiswise
```

This promotion is limited to the generic transform and field-storage layers.
Plane and Channel therefore inherit the new requested policies when they use
the generic solver defaults, but their spectral-storage choices, equations,
boundary-condition semantics, spectral differentiation, dealiasing, and PDE
models do not change. Mixed-boundary legacy-order transforms retain their
guarded axiswise fallback, and non-contiguous transform groups continue to
fall back to advanced indexing.

Qualification is deliberately split into two stages. The opt-in A/B/C/D stage
establishes numerical agreement, performance, and memory behavior for each
factor and their combination. The default-path smoke then omits both selectors
and requires the implicit result to be bitwise identical to the explicit
combined candidate. It also verifies that the historical rollback remains
available and numerically agrees within a dtype-aware relative-L2 tolerance.

Run the default-path smoke from the repository root as documented in
`benchmarks/README.md`. Preserve its JSON output with the Git commit and GPU
provenance when qualifying a release or integration branch.
