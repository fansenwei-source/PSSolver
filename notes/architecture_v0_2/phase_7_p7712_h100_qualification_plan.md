# Phase 7 P7.7.12: public simulation H100 non-regression plan

Status: `CPU_ORACLE_RECOVERY_READY_H100_PENDING`.

P7.7.12 is the final P7.7 closure. Its machine-readable authority is
[phase_7_p7712_h100_qualification_plan.json](phase_7_p7712_h100_qualification_plan.json).
The frozen scientific baseline is P7.7.11 commit `b90467f`; this stage adds
qualification tooling and evidence only.

The first HPCC CPU gate stopped before submitting an H100 job because the
P7.7.11 test incorrectly treated a desktop software build's raw NPY hashes as
cross-environment numerical oracles. The recovered contract is documented in
[phase_7_p7712_cpu_oracle_recovery.json](phase_7_p7712_cpu_oracle_recovery.json):
absolute fingerprints remain exact for their matching environment, while
every environment must independently pass strict direct/public and
continuous/restart byte identity. No numerical tolerance or solver change was
introduced. The formal H100 submission allowance remains unused.

## What is being compared

For each already-qualified application/runtime pair, one immutable public
`Simulation` is compiled and then executed through either:

1. the pre-existing Plane or Channel application entry (`direct`); or
2. public `run_simulation()` (`public`).

Both routes consume the same compiled application request. Selection occurs
before allocation and there is no dispatch or wrapper work in the timestep.
The matrix contains Plane `legacy_production` and `compiled_v2`, plus Channel
`legacy_channel` and `compiled_channel_v2`.

The H100 job uses representative R128 grids. It deliberately reuses the
P7.7.6 large-grid H100 closure instead of repeating its R320 Plane and R512
Channel package-runtime matrix. P7.7.6 already qualified the underlying
production applications, performance, memory, restart, and negative gates;
P7.7.12 isolates only the public declaration/compiler/runner/result layer
added afterward.

## Frozen execution matrix

Each of the four cells runs three balanced direct/public 60-step pairs. Every
pair must have byte-identical final Q, velocity, pressure, diagnostics, and
COMPLETE artifacts. Requested and effective runtime paths must agree, runtime
fallback must be false, every output must be finite, TF32 must remain off, and
all reports must identify one NVIDIA H100 PCIe.

The public/direct ratio limits are 1.05 for the arithmetic mean and median
workflow timestep, 1.01 for peak allocated memory, and 1.01 for peak reserved
memory. These are non-regression limits, not claims that the wrapper accelerates
the numerical kernel.

For restart, the public entry writes a 30-step checkpoint. Direct and public
entries independently resume that checkpoint for 30 further steps. Their step
60 artifacts must be byte-identical to one another and to the corresponding
uninterrupted 60-step pair. Cross-runtime restart is outside this stage because
the applicable rejection gates already passed in P7.7.6.

## Fail-closed execution

The repository-owned case runner records Git, CUDA, runtime, result, memory,
checkpoint, and artifact identities. The repository-owned analyzer requires
exactly 36 reports and rejects missing, duplicate, dirty, fallback, non-finite,
wrong-device, wrong-clock, numerically unequal, restart-unequal, or
performance-regressed evidence.

The login-node CPU gate deselects exactly the three CUDA-only tests named in
the machine plan. Those tests must then pass, without skip or deselection,
inside the single H100 job before any scientific case starts. No automatic
retry or second formal H100 submission is allowed.

The repository-local qualification support passed 14 targeted tests and the
complete suite of 2189 tests plus 8 subtests, with no failure, skip, or
deselection. Real CPU smoke runs also exercised a Plane direct entry and a
Channel public entry with checkpoint creation. These checks qualify the
runner/analyzer mechanics only; they do not replace the pending H100 gate.

## Scope boundary

This stage changes no numerical kernel, mathematical model, boundary
condition, checkpoint schema, runtime default, or production promotion. It
does not run a long-time scientific benchmark and does not authorize Phase 8
or Phase 9 before the H100 evidence is analyzed and committed as a separate
closure record.
