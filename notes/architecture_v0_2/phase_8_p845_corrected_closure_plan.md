# P8.4.5 corrected numerical and memory closure contract

Status: `READY_P8_4_5_CORRECTED_CONTRACT_CLOSURE`.

## Evidence requiring correction

The original P8.4.5 contract treated two implementation-dependent quantities
as scientific identities: CUDA allocator reserved bytes and cross-version
byte identity after changing a compiled pointwise graph.  Jobs `10843818` and
`10843883` provide enough evidence to correct those categories without
relaxing the scientific problem.

For the three-step parent/candidate comparison, eager execution was exactly
byte-identical and all five compiled Q components remained byte-identical.
Only compiled `p`, `ux`, `uy`, and `uz` differed.  Their largest fieldwise
relative L2 was `4.963388905214048e-16` and the largest absolute difference was
`4.336808689942018e-19`.  Candidate continuous/restart trajectories remained
byte-identical.  This is compiled floating-point operation-boundary roundoff,
not a changed equation, state representation, transform count, or restart
contract.

At R128 the lifting runtime added `21,106,688` peak active/allocated bytes but
`138,412,032` reserved bytes.  Only about 15.15% of the reserved delta was live
allocation.  R320 simultaneously had positive active allocation and negative
reserved delta.  Reserved bytes therefore measure allocator cache and size
classes rather than persistent scientific storage.  The implementation owns
no persistent full-domain lift, affine Laplacian, linear correction, or
physical-Q workspace.

## Corrected contract

The final closure retains all existing wall, manufactured, finite-state,
transform-count, graph-break, timestep, package, and provenance gates.  It
changes only the two misclassified identities:

- peak allocated and peak active memory each retain the unchanged `1.10`
  lifting/control ceiling;
- peak reserved memory remains mandatory evidence but is diagnostic-only;
- same-runtime continuous/restart state remains byte-for-byte exact;
- cross-version compiled execution uses per-field relative L2 and absolute
  Linf limits of `1e-12` after at least 20 matched steps from identical Q;
- ULP and mismatch counts remain diagnostics and cannot replace L2/Linf;
- no tolerance is applied to checkpoint integrity, runtime identity, initial
  state identity, finite-state checks, or transform counts.

This is not a retroactive pass for Job `10843818`.  A new installed-package
H100 closure must produce active-memory fields, a fresh 12-profile matrix, the
manufactured oracle, exact same-runtime restart, and one repository-adjudicated
20-step cross-version comparison.

P8.5, nonhomogeneous Neumann data, Phase 9, compiled-runtime promotion, and
production-default changes remain unauthorized.
