# V3 initialization protocol for the Shendruk Fig. 4 benchmark

V3 denotes the following target-parameter initialization protocol:

1. For each unique physical pair `(frank_k, zeta)`, evolve a strictly
   `z`-independent two-dimensional state with the same Beris--Edwards equation,
   complete nematic stress, precision, and dealiasing used by the 3D run.
2. Qualify a late time window as a candidate 2D statistical steady state.
3. Bind one representative Q2D checkpoint, its SHA-256, the qualification
   report, and the exact `(frank_k, zeta)` pair in a V3 mother manifest.
4. Extrude that Q2D checkpoint through the target height and apply a small
   local rotation field with periodic `x,y` modes and strictly positive
   Neumann-compatible wall-normal DCT modes.
5. Evolve the 3D system from the first step with the same target `(frank_k,
   zeta)` used to generate the mother. There is no activity quench in V3.

The protocol is designed to initialize straight disclination lines sampled
from already established quasi-2D active turbulence. It is a reproducible
independent reconstruction because the Shendruk paper does not publish a
point-by-point initialization recipe.

## Stage A: target-parameter Q2D mother

Use the production model script with `v3-mother`. A small wall-normal grid is
an efficiency device only after its height-independence has passed the
repository tests and an HPCC GPU preflight.

```bash
python Plane_beris_edwards_stokes.py \
  --activity-number 18 \
  --height 20 \
  --parameterization paper-window \
  --output-dir /NEW/EMPTY/MOTHER_DIR \
  --initialization-protocol v3-mother \
  --v3-mother-noise-rms 0.01 \
  --v3-mother-max-mode-x 4 \
  --v3-mother-max-mode-y 4 \
  --nx 320 --ny 320 --nz 4 \
  --dt 0.005 --steps MOTHER_STEPS \
  --save-start-step MOTHER_SAVE_START \
  --save-interval 1000 --diagnostic-interval 1000 \
  --dtype float64 --tf32 off \
  --dealias-rule cubic_half \
  --zero-mode-policy zero_mean \
  --disable-spectral-refresh \
  --device cuda
```

The run writes numeric `Q2D_<step>.npy` frames, online mother diagnostics, and
`z_invariance.json`. It aborts rather than save a Q2D frame if the configured
relative `z`-invariance tolerance is violated.

If the target mother needs more time than one walltime, continue it without
reapplying noise. Saved and diagnostic step numbers remain absolute:

```bash
python Plane_beris_edwards_stokes.py \
  --activity-number 18 --height 20 --parameterization paper-window \
  --output-dir /NEW/EMPTY/CONTINUATION_DIR \
  --initialization-protocol v3-mother \
  --v3-mother-restart-q2d /PREVIOUS_DIR/Q2D_START_STEP.npy \
  --v3-mother-start-step START_STEP \
  --nx 320 --ny 320 --nz 4 \
  --dt 0.005 --steps ADDITIONAL_STEPS \
  --save-start-step START_STEP --save-interval 1000 \
  --diagnostic-interval 1000 --dtype float64 --tf32 off \
  --dealias-rule cubic_half --zero-mode-policy zero_mean \
  --disable-spectral-refresh --device cuda
```

Continuation currently requires spectral refresh to remain disabled so its
phase cannot silently reset. The restart checkpoint is rehashed before
`COMPLETE` is written.

## Stage B: conservative mother qualification

Select a late window fixed before looking at its result:

```bash
python scripts_plane/qualify_v3_2d_mother.py \
  --run-dir /MOTHER_DIR \
  --run-dir /OPTIONAL_CONTINUATION_DIR \
  --output-dir /NEW/EMPTY/QUALIFICATION_DIR \
  --start-time WINDOW_START \
  --end-time WINDOW_END
```

Exit code 0 means the finite-window candidate gates passed. Exit code 2 means
the analysis completed but the mother is not qualified. The gate checks a
dynamic neutral defect population, defect-count-change birth/death proxies,
ESS, full-window drift, half-window compatibility, and z invariance. It is a
conservative finite-window screen, not a proof of ergodicity.

## Stage C: immutable V3 mother manifest

Only a qualification report with
`candidate_2d_statistical_steady_state=true` and a recommended checkpoint can
produce a manifest:

```bash
python scripts_plane/create_v3_mother_manifest.py \
  --checkpoint /MOTHER_DIR/Q2D_SELECTED_STEP.npy \
  --qualification-report /QUALIFICATION_DIR/v3_mother_qualification.json \
  --frank-k TARGET_K \
  --zeta TARGET_ZETA \
  --output /NEW/V3_MOTHER_MANIFEST.json
```

## Stage D: target-parameter 3D run

```bash
python Plane_beris_edwards_stokes.py \
  --activity-number TARGET_A \
  --height TARGET_H \
  --parameterization paper-window \
  --output-dir /NEW/EMPTY/THREE_DIMENSIONAL_DIR \
  --initialization-protocol v3-extruded \
  --initial-q2d /MOTHER_DIR/Q2D_SELECTED_STEP.npy \
  --v3-mother-manifest /V3_MOTHER_MANIFEST.json \
  --v3-rotation-rms 0.001 \
  --v3-rotation-max-mode-x 3 \
  --v3-rotation-max-mode-y 3 \
  --v3-rotation-z-modes 1 2 3 \
  --v3-perturbation-seed PERTURBATION_SEED \
  --nx 320 --ny 320 --nz TARGET_NZ \
  --dt 0.005 --steps 40000 \
  --save-start-step 20000 --save-interval 1000 \
  --dtype float64 --tf32 off \
  --dealias-rule cubic_half \
  --zero-mode-policy zero_mean \
  --disable-spectral-refresh \
  --device cuda
```

The runner refuses to start if the checkpoint, qualification report, manifest,
`K`, or `zeta` do not match. It rehashes all three bound input files before
writing `COMPLETE`.

## Pairing and reuse

- A Q2D mother belongs to a physical `(K, zeta)` pair, not to `A` alone.
- The same mother ensemble should be reused across heights and boundary
  conditions whenever `(K, zeta)` is identical.
- Distinct mother snapshots and 3D perturbation seeds are needed for ensemble
  uncertainty.
- The main V3 curve must use target-matched mothers. A high-activity mother
  followed by an immediate target-activity quench is a separate sensitivity
  protocol, not V3.
