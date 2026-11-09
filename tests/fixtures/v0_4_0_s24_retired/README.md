# S-24 retirement fixtures

Authentic one-step CPU float64 Plane checkpoints written by the installed
v0.4.0 wheel, before S-24 removes the canary implementation. Neither metadata
nor tensors were modified after writing. `private_lifting` supplements the
older rc1 lifting fixture; its private runtime remains deferred to S-27.

- Source: `4c12228aba01e8483e80d28a1471ab72384d5fda`.
- Wheel SHA-256: `c5cf3e26af8bdde27287f6129fe5939d6f202fb627dcdc9a66c19d62871dd873`.
- Producer: `qualification/generate_s24_retired_fixtures.py`.
- Producer SHA-256: `2e85fca9be0b0ae8d19d49582f11f1c4a7e809cad5a3edf01f12a95ee2be3136`.
- Canary metadata: `8cb2d22c6ce0bc38ee5730a843ac5df8dd64b65965ebe74feec199308d5add52`.
- Lifting metadata: `1263094dab60e6d5b39bd06eeef7050481ca689d7616b0b5deb0cd25b29f6ce3`.

The sealed checkpoint metadata binds every tensor's size, shape, dtype and
SHA-256. S-24 must reject both families before reading any tensor, with an
explicit instruction to use v0.4.x / maint/v0.4. Historical file paths in the
authentic metadata are provenance, not paths to be accessed by tests.
