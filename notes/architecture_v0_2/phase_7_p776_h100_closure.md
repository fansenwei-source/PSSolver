# Phase 7 P7.7.6: production-connection qualification closure

Status: `P7_7_6_COMPLETE_PHASE_8_PLANNING_ELIGIBLE_NOT_AUTHORIZED`.

Candidate: `174cf4df893e0fa1af713349f11e73b55ff9c8bb` on
`next/pssolver-v0.2.0-architecture`.

Baseline: `1078f44214d03fdce9326955f176e17aa36c0093`.

Final classification:
`PASS_P7_7_6_PRODUCTION_CONNECTION_H100_NON_REGRESSION_WITH_TARGETED_ADJUDICATION_AND_FINAL_CPU_GUARD_RECOVERY`.

This record closes the H100 qualification required by the local P7.7.6
candidate record.  It is additive: the earlier
`phase_7_p776_production_connection` files continue to describe the state at
candidate creation, when H100 qualification was still pending.  They are not
rewritten after the fact.

The machine-readable authority is
[phase_7_p776_h100_closure.json](phase_7_p776_h100_closure.json).

## Result

The production application connection is qualified for all supported paths:

- Plane `legacy_production`;
- Plane `compiled_v2`;
- Plane `separated_canary`, retained as an explicit compatibility exception;
- Channel `legacy_channel`;
- Channel `compiled_channel_v2`.

The four package-qualified paths construct from a complete
`PackageRuntimeConstructionInput`.  Requested and effective runtime identities
match, fallback is false, and the package plan kind matches the selected path.
The separated canary remains outside the package binding, records a null
package plan, and continues through its historical facade without fallback.

## CPU and H100 evidence

The candidate passed 181 targeted CPU tests and the HPCC full suite with 2,120
passing tests, one allowed optional-dependency skip, three authorized
CUDA-only deselections, and eight passing subtests.  The three deselected tests
then passed on the allocated NVIDIA H100 PCIe.

The original H100 matrix produced 48 complete profile JSON files.  Seven of
eight cells satisfied the preregistered three-percent timing threshold.  The
only initial failure was the smallest Channel compiled cell.  It had identical
initial and final state hashes, identical memory, correct 11/36 steady-state
transform calls, no fallback, and no graph breaks, but three short pairs gave
a noisy ratio slightly above the threshold.

A preregistered twelve-pair adjudication used 20 warm-up steps and 200 measured
steps per independent process.  Its final ratios were:

```text
arithmetic mean B/A:  1.0039120579297005
median B/A:           1.0151954952542155
geometric mean B/A:   1.004142344359126
bootstrap 95% CI:     [0.9845594354899496, 1.0248376844826659]
B faster:              7/12
allocated ratio:       1.0
reserved ratio:        1.0
```

The recovered measurement is inside the unchanged mean and median threshold
of 1.03.  It establishes performance equivalence; no threshold was relaxed
and no paired-win condition was introduced after observing the result.

## Numerical, restart, and rejection closure

Matched 100-step baseline/candidate trajectories for all five paths produced
byte-identical `Q`, `u`, and `p`.  Forty continuous, segmented, and resumed
output directories carried valid `COMPLETE` markers.  Same-runtime restart,
baseline-to-candidate restart, and candidate-to-baseline restart matched their
continuous references byte for byte.

The final evidence also covers:

- Plane and Channel cross-runtime rejection before mutation;
- rejection of treating `separated_canary` as a package runtime;
- legacy and compiled Channel runtime-identity tamper;
- tensor raw-byte and checksum-metadata tamper;
- backend restart-contract tamper reaching the intended backend guard;
- file-record shape tamper reaching the intended shape-record guard;
- physical, spectral, pressure, progress, storage, and workspace invariance
  for every applicable pre-mutation rejection.

The last two checks used the same-spec in-memory checkpoint construction and
disk-record mutation semantics already frozen by the repository's authoritative
Channel checkpoint tests.  The final CPU-only Job completed successfully.

The imported closure record itself passed 12 targeted local audit tests.  The
complete repository suite then passed with 2,130 tests and eight subtests.

## Evidence chain

Four immutable HPCC archives form the closure chain:

1. the original H100 matrix and its fail-closed performance result;
2. the twelve-pair performance adjudication plus trajectories and restart;
3. the first Channel negative-gate recovery, retained as a failed harness
   attempt;
4. the final backend/file-record recovery with a valid `COMPLETE` marker.

All four checksum manifests were revalidated after the final Job.  Their
entry counts and manifest identities are frozen in the JSON record.  Failed
intermediate jobs remain part of the provenance; they are not presented as
solver, numerical, scientific, CUDA, or performance failures.

## Authorization boundary

P7.7.6 is complete and Phase 8 planning is eligible.  Phase 8 execution is not
authorized by this result.  Phase 9 is also not authorized.  Production
defaults remain `legacy_production` for Plane and `legacy_channel` for Channel,
and no compiled runtime is promoted.

This closure changes no numerical code, runtime default, checkpoint schema,
or public API.  It only records the qualification evidence for candidate
`174cf4df893e0fa1af713349f11e73b55ff9c8bb`.
