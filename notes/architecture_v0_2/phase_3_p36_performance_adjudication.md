# Phase 3 P3.6 H100 performance adjudication

Status: `PASS_SIMULATION_TIMER_NON_REGRESSION`.

Phase 2 closure baseline:
`0d5b7186b7a104354db173e04a43e68c15daa515`.

Phase 3 cumulative candidate:
`227b095c12e494ca74756228a30962692474c516`.

Qualification evidence commit:
`503d794dc7bea9ace11d23b60df95805b00037bb`.

The cumulative Phase 3 candidate passed an independent, pre-registered H100
simulation-timer non-regression adjudication at R128 and R320.  This resolves
the P3.6 performance gate and authorizes the remaining closure continuation.
It does not by itself close Phase 3, change the production default, promote
`separated_canary`, or authorize Phase 4.

The machine-readable record is
[phase_3_p36_performance_adjudication.json](phase_3_p36_performance_adjudication.json).

## Why an independent adjudication was required

The initial cumulative P3.6 task stopped at its frozen three-trial R128
profiler gate.  The observed coefficient of variation was about four percent,
larger than the two-percent non-regression threshold, while R320 was stable and
passed.  A source audit also established that the direct-component profiler
does not execute the runtime selector or the Phase 3 production facade.

The earlier result remains recorded as `FAIL_P3_6_CLOSURE`; it was not
retroactively changed.  A separate task therefore pre-registered the
production workflow's own simulation timer as the primary metric and
generated an independent data set.  External whole-process wall time remained
a secondary deployment-latency measurement because it includes interpreter
startup, imports, compilation, initialization, and output handling.

## Frozen measurement contract

The primary per-run metric was

```text
metadata.elapsed_seconds / metadata.workflow.requested_additional_steps
```

For each grid, twelve balanced A/B pairs were collected through the formal
`Plane_beris_edwards_stokes.py` production entry with
`runtime_path=legacy_production`, float64 arithmetic, TF32 disabled, identical
initialization and numerical options, independent caches, and no periodic
snapshots.  R128 used 1000 steps per process and R320 used 500.

The pre-registered statistic was the paired geometric mean of candidate over
baseline seconds per step.  A 200,000-sample paired log-ratio bootstrap with
seed 20260921 produced a two-sided 95-percent percentile confidence interval.
Passing required both the point estimate and confidence-interval upper bound
to be no greater than 1.02.  No sample was replaced, removed, or winsorized.

## H100 result

The single formal job, 10836439, completed on one NVIDIA H100 PCIe with exit
code 0:0.  All 48 production-entry processes completed and passed their
runtime, adapter, fallback, completion, shape, dtype, finite-value, artifact
identity, derived-value, and command-record round-trip gates.

| Grid | Baseline mean s/step | Candidate mean s/step | Paired geometric B/A | 95% CI | Result |
|---|---:|---:|---:|---:|---|
| R128 | 0.005760753751 | 0.005787593186 | 1.004657429 | [1.000614770, 1.008715823] | PASS |
| R320 | 0.057287097295 | 0.057294978340 | 1.000136884 | [0.999480996, 1.000810543] | PASS |

Both confidence-interval upper bounds are below 1.02.  Consequently,
`performance_non_regression_resolved=true`, no performance optimization is
required, and P3.6 closure continuation is eligible.

## Derived-value contract

Seconds per step is derived from the two authoritative metadata fields rather
than compared with a hand-transcribed decimal.  The validator records the raw
fields, Python representation, 17-digit representation, and IEEE-754
`float.hex()` value, then verifies command-record round trips.  Its synthetic
suite passed 33 tests, including zero or negative denominators, NaN, Inf,
non-integer step counts, and a one-ULP tamper.

## Evidence

The authoritative evidence directory is
`/home/fansenwei/pssolver_phase3_p36_simulation_timer_adjudication_503d794_20260920_recovery_v3`.
Job 10836439 completed in 00:27:55 with requeue disabled and zero restarts.
The checksum manifest passed 4228/4228 entries and has SHA-256
`ce19ad4a1d3dbc8840a3bda59896a7ba9246ed145ef5ae2dae33c0317b95df46`.

The principal report hashes are:

| Artifact | SHA-256 |
|---|---|
| `final_report.md` | `33ce8b17c83614f87392ffec55e09ff907df13b69d5c8b90db6d0d0f7f70b48d` |
| `summary.json` | `8acb7581ec51491a6a5afc920b528a46466777833b1dab4d3bae60e81a77a6f7` |
| `summary.csv` | `4ab888daee13ad77a5e56556d3101e625bf3b86373131dc97cf2994a0d1b76bb` |
| `derived_value_contract.json` | `ce15dd0c28ba6e6c8b7b204cf67670d0df90432585575f5463d04618a4fe3be0` |
| `real_fixture_validation.json` | `35966fe3c30c79709c21257d3c9b7ef62ff1eaf9b70180347262f4707951d61b` |
| `command_record_round_trip.json` | `a4de7e58abc6ea41e1dc004738404386621937240b8d01f49c2a304395429fc3` |

## Remaining P3.6 closure work

Performance measurement is complete and must not be repeated during the
closure continuation.  Phase 3 remains open only until the cumulative
continuous-trajectory, bidirectional restart, reset/rebind, and negative-gate
evidence is completed and archived.  Passing those remaining gates may close
Phase 3 and authorize Phase 4 planning, but it still does not authorize Phase
4 execution or any runtime/default promotion.
