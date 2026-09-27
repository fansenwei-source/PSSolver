# Phase 8 P8.4.5: CUDA allocated-device identity recovery

Status: `READY_P8_4_5_CUDA_DEVICE_IDENTITY_RECOVERY`.

The first P8.4.5 H100 job, 10843514, passed Git, CPU, installed-package,
CUDA-environment, and CUDA-only test gates.  It completed the first
homogeneous R128 control profile, then stopped while constructing the first
strong-planar lifting runtime.  No lifted timestep or timing window started.

The source failure was a device-identity mismatch.  The lifting operator kept
the unindexed allocation request `torch.device("cuda")`, while materialized
tensors correctly reported the concrete device `torch.device("cuda:0")`.
Direct equality between those objects is false, so strict initial-field
validation rejected an otherwise correctly allocated tensor.

The recovery keeps strict device checking but changes its authority.  The
operator uses the requested device to allocate the lift, then records the
actual materialized lift tensor's device as its runtime identity.  All input,
output-workspace, restart, and provenance checks therefore compare concrete
allocated identities.  CPU behavior is unchanged, and no cross-device copy
or permissive type-only comparison is introduced.

A CUDA regression test constructs the real strong-planar Plane runtime using
the unindexed public device spelling `cuda` and requires lifting provenance
to bind to the current concrete CUDA device.  The complete P8.4.5 matrix must
be rerun in a new output directory; the one old control profile is not reused.

This recovery does not change the lifting equation, wall data, numerical
thresholds, runtime selection, production defaults, or the P8.4.5 analyzer.
It does not authorize nonhomogeneous Neumann data, P8.5, or Phase 9.
