# Architecture Stage Q.6.2 closure: native segments rejected

Stage Q.6.2 tested the Q.6.1 `plane_native_segment_handoff_shadow_candidate`
on an NVIDIA H100 PCIe.  The candidate was experimental Plane-shadow code and
never changed a production default, the generic solver, Channel, equations,
boundary signatures, or integration semantics.

## Formal evidence

- Candidate commit: `e69cdd38e379951009e4bc43b7edf7c203ed133e`
- Analysis-contract recovery commit:
  `705b93fc51cdaf6f0697068fc024844430de7f93`
- H100 Job: `10832425`
- CPU-only formal recovery Job: `10832469`
- Formal classification: `C_rejected`
- Recovery archive manifest SHA-256:
  `b131776c8a44eb986b44b6b2a76a531e65c3a8086695fef0146dcb192f55bebe`

The formal result is archived at:

`/home/fansenwei/pssolver_stage_q62_analysis_recovery_705b93f_20260916_v1`

## Results

The candidate eliminated `copy_cat` at all three required sources and used no
workspace or retained tensor reference.  The six-step Q/u/p comparison passed
with maximum relative L2 error `3.474708366745567e-15`, below the frozen
`1e-12` threshold.  Peak allocated and reserved memory ratios were
`1.0000483888998284` and `0.9948247078464106`, respectively.

The resulting segments were nevertheless all singletons.  The candidate
created 34 native-segment transform batches per timestep, including 29 extra
transform batches.  Across three balanced R320 trials the baseline mean was
55.167460569 ms per timestep and the candidate mean was 64.057560730 ms.  The
candidate/baseline ratio was `1.161147532795284`; the candidate was slower in
all three paired trials and failed both the performance and 1.05 safety gates.

## Disposition

The scheduler-level native-segment candidate is closed and must not proceed to
Q.6.3 or production promotion.  Its runtime implementation and qualification
scripts were removed after formal closure.  Git history and this note retain
the negative result.  Any future revisit requires producers to emit genuinely
multi-component, boundary-signature-native storage; scheduler-side singleton
partitioning must not be reintroduced as an optimization.
