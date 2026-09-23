# Phase 7 P7.1: Channel run-spec decomposition

Status: `PASS_P7_1_CHANNEL_RUN_SPEC_DECOMPOSITION`.

P7.1 introduces a tensor-free, immutable Channel declaration graph and a flat
provisional compatibility facade at its direct module path.  It is not added
to the frozen `pssolver.configuration` package-root export surface and does not
connect the new declarations to
`Channel.py`, `pssolver.channel`, a workflow, or a GPU runtime.  Consequently
the legacy Channel operation order, output schema, numerical results, and
defaults are unchanged.

The flat `ChannelActiveNematicRunSpec` contains 37 public initialization fields
plus one private cached component graph.  Its defaults characterize the
current `Channel.py`: shape 512x40x40, lengths 128x10x10, dt 0.01, 1,000
steps, rho 6, K 1, activity 5, beta -1, zero friction, unit viscosity,
float32, no dealiasing, real-first full-complex transforms, generated aligned-x
initial Q, and the existing save and diagnostic schedules.

The decomposition gives separate owners to geometry, homogeneous boundary
spaces, numerics, material parameters, Stokes request, PCG pressure controls,
initialization, execution identity, and workflow/output policy.  The Stokes
declaration explicitly records a zero-mean pressure gauge and a
`not_applicable` tangential zero-mode policy: no-slip walls remove the Plane
plug-flow ambiguity.

Only `legacy_channel` is a valid runtime selector in P7.1.  A request for
`compiled_channel_v2` fails closed because that runtime has not been built or
qualified.  Likewise, alternate boundary spaces, dealiasing, projected
transform execution, or spectral storage are rejected rather than advertised
as supported behavior.

The declarations import no Torch module and allocate no tensor or solver.
SHA-256 tests confirm that `Channel.py` and `pssolver/channel.py` remain
byte-identical to the P7.0 oracle inventory.  The new metadata is explicitly
marked `production_connection=false` and has a deterministic canonical hash.
The single new configuration-to-systems dependency is named explicitly in the
architecture import ratchet.

P7.1 makes P7.2 planning eligible.  It does not authorize extracting or
replacing the Channel Stokes/Schur solver, creating a compiled runtime,
changing a default, running H100 qualification, adding new boundary laws, or
moving optimal-control code.

Local validation passed 125 targeted and compatibility tests and the complete
suite passed 1,979 tests plus 8 subtests with no failure.

The machine-readable result is
[phase_7_p71_run_spec_decomposition.json](phase_7_p71_run_spec_decomposition.json).
