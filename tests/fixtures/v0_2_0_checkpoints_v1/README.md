# PSSolver 0.2.0 checkpoint fixtures

This directory is an immutable compatibility corpus written by the published
`pssolver-0.2.0-py3-none-any.whl` artifact (SHA-256
`ab4beef1a347568e8ad4162291feb62e94e179849853ca1417583634649a1151`).
It contains one-step checkpoints for the two stable functional runtimes and
the four production checkpoint families present in 0.2.0:

- `functional_periodic`
- `functional_channel`
- `production_periodic`
- `production_channel_complete_stress`
- `production_plane_legacy`
- `production_legacy_channel`

`manifest.json` authenticates every file and records the raw tensor digests
expected after one additional timestep. Compatibility tests must read these
directories in place, must not rewrite or upgrade them in place, and must
produce the registered continuation state bit-for-bit. A future format that
cannot directly import a fixture must use an explicitly documented,
non-in-place migration and retain the original fixture unchanged.

The fixture generator is
`benchmarks/v0_2_0_checkpoint_fixtures.py`. It was executed outside the
source checkout with the published wheel installed; the imported package path
is recorded in the manifest. Regeneration is not part of normal testing.
