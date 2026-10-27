# Historical CPU coverage fixtures

These fixtures close the v0.3 S-08 coverage gaps with checkpoints written by
the exact released rc1 and rc2 source commits.  They are not hand-authored or
rewritten copies.

- `rc1_channel_functional_odd` and `rc2_channel_functional_odd` use a Channel
  functional runtime whose periodic axis has length seven.  Each directory
  contains the released checkpoint, its construction snapshot, and the next
  state written by the same released runtime as a continuation oracle.
- `rc1_plane_lifted` is a real rc1 Plane format-1 checkpoint written by the
  static-lifting runtime with non-null `lifting_restart` metadata and odd
  periodic axes.

`manifest.json` binds the generator, release commits, reproducible git
archives, generation environment, and every case tree.  Regeneration must use
an exact released source tree first on `PYTHONPATH`; generated checkpoints are
immutable test evidence.  The generation environment includes the same
`numerical_build_fingerprint` used by the v0.2.0 continuation corpus.  A
matching build requires bitwise rc2 continuation; another build must remain
within `8 * dtype epsilon`.  The rc1 state-only difference must exceed that
roundoff allowance by at least a factor of one thousand.

S-10 advances that fingerprint to schema 2. Torch binds its explicit version,
CUDA version, git version, and sole `Build settings:` payload, without relying
on visibility-dependent CUDA/CuDNN probe lines or their presentation format.
NumPy uses the structured build dictionary when available and a captured
legacy `show()` representation on NumPy versions that lack `mode="dicts"`;
the selected source is recorded. The checkpoint, snapshot, and continuation
payloads remain unchanged.
