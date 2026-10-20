# PSSolver v0.2.1 local release preparation

Date: 2026-10-20

Classification: `PASS_V0_2_1_RELEASE_SOURCE_PREPARATION`

This is a narrowly scoped maintenance release over immutable `v0.2.0`.  Its
only runtime change is the already qualified N-1 NFS-compatible, fail-closed,
no-clobber checkpoint publication path.  The public version is advanced from
`0.2.0` to `0.2.1`.

The solver equations, numerical algorithms, checkpoint bytes and schema,
Functional API version 1.0, provider identity, production defaults, and
qualified model--geometry combinations are unchanged.  The immutable
`v0.2.0` tag and assets are not rewritten.

N-1 passed local fault-injection testing, installed-wheel CPU testing, real
NFS testing on both `$HOME` and `/scratch1`, two-client collision testing, and
the small installed-wheel H100 checkpoint/restart qualification in Job
`10902665`.  Periodic and Channel functional runtimes both matched continuous
execution byte for byte across state, Q, velocity, and pressure.

This record authorizes only local release preparation.  Push, tag creation,
GitHub Release creation, PyPI publication, and another H100 submission remain
unauthorized.  Clean source archive, sdist, and wheel identities are recorded
separately after they are built from this exact release source.
