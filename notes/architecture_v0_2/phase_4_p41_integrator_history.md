# Phase 4 P4.1 integrator declaration and disconnected history

Status: `P4_1_LOCAL_COMPLETE_DISCONNECTED`.

P4.1 adds two provisional direct-import modules:

- `pssolver.core.integrators` declares tensor-free projected
  semi-implicit-Euler and constant-step SBDF2 identities;
- `pssolver.integrators.history` owns one immutable, complete SBDF2 history
  level in native spectral storage.

The declaration fixes projected semi-implicit Euler as the SBDF2 startup
scheme, formal order two, constant `dt`, and history depth one.  The history
stores the previous evolved native spectrum and previous explicit native-
spectral RHS by caller-supplied tensor identity.  It performs no clone,
conversion, transform, allocation, runtime registration, or import-time
connection.

History validates shape, dtype, device, storage separation, source step, and
exact `dt` compatibility with a current spectrum.  It deliberately does not
claim tensor finiteness or checkpoint integrity; those belong to the later
checkpoint and negative-gate slice.

Neither module is re-exported from `pssolver`, `pssolver.core`, or
`pssolver.integrators`.  No runtime imports the new declarations or history,
the Plane checkpoint-v1 implementation is unchanged, and no production
default changes.  P4.2 is authorized only after this local disconnected
contract passes its complete CPU regression.

Local qualification passed 29 focused architecture/history tests and the
complete suite of 1,685 tests plus 8 subtests.  The archive source inventory,
JSON validation, and whitespace gate also passed.

The machine-readable record is
[phase_4_p41_integrator_history.json](phase_4_p41_integrator_history.json).
