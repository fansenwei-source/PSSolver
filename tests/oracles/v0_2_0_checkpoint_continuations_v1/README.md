# PSSolver 0.2.0 continuation oracle

This directory is intentionally separate from the immutable checkpoint corpus
in `tests/fixtures/v0_2_0_checkpoints_v1`.  It stores the one-step continuation
arrays produced by the published PSSolver 0.2.0 wheel in the numerical build
recorded by `manifest.json`.

Checkpoint import is always required to reproduce every stored payload bit for
bit.  Continuation is required to be bitwise identical only when the complete
numerical-build fingerprint matches the reference build.  Other PyTorch/FFT
builds use the recorded arrays with the fail-closed `8 * dtype epsilon`
absolute and relative tolerance.  The selected mode and both build
fingerprints are returned by `replay_case`.

The reference fingerprint is measured in the actual Nematics3D conda-forge
environment that generated this oracle.  Schema 2 binds Torch's explicit
version, CUDA version, git version, and sole `Build settings:` payload; it does
not hash visibility-dependent CUDA/CuDNN probe lines.  NumPy records whether
its structured `mode="dicts"` API or the captured legacy `show()` text supplied
the build configuration.

The oracle was generated without modifying the checkpoint corpus:

```text
python benchmarks/v0_2_0_checkpoint_fixtures.py \
  --continuation-oracle \
  --fixture-root tests/fixtures/v0_2_0_checkpoints_v1 \
  --output tests/oracles/v0_2_0_checkpoint_continuations_v1
```
