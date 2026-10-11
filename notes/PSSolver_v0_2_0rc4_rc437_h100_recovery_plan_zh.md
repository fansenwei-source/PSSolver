# PSSolver RC4.3.7 H100 correct-layer recovery 计划

Job `10874707` 已按冻结合同停止，旧归档保持失败状态且不得改写。该 Job 已通过分解式 H100/TF32/host-compiler 预检、7 个 CUDA-only tests、CPU/CUDA conditioning relative-L2、wall residual、transition change、fresh provenance，以及 4-step continuous 对 2+restore+2 的逐字节 restart 门禁；12 个 profile 尚未启动。

停止项不是求解器实现错误。旧 CUDA helper 在 `adapter.fields.spatial` 上检查了 localized conditioning 的 interior byte identity；该对象已经经过全局 truncated spectral projection。壁面附近的局部修改经过全局谱投影后可以影响整个重构域，因此这个层级不能验证 conditioning policy 的局部性。

权威门禁层改为 `PlaneStaticLiftingRuntime.extract_initial_remainders(..., apply_fresh_conditioning=True)` 的直接输出，也就是进入全局谱投影之前的 homogeneous remainder。v2 helper 将在 CPU 和实际 `cuda:0` 上分别构造 direct conditioned/unconditioned remainder，要求五个 Q 分量的 interior 逐字节一致、transition region 确实改变，并继续要求 CPU/CUDA relative L2 不超过 `1e-15`。投影后的 adapter-field 比较仍会记录，但只作诊断，不能参与 PASS/FAIL。

旧 v1 helper、analyzer、计划和 Job 归档均保持原样。recovery 只新增 v2 qualification helper 与 analyzer，并扩充合成门禁；没有修改 runtime source、数值方程、谱方法、阈值、checkpoint identity、公共 API、生产默认值、PSSolver-Control 或 nematics3d。

新的 H100 recovery 必须重新生成 candidate CUDA correctness 证据和 12 项 A/B profile matrix，不能把 Job `10874707` 重新解释为 PASS。它需要用户另行明确授权，只允许一个正式 H100 Job，且不得自动重试。

本地 recovery support 已通过 69 项定向测试。完整 CPU suite 为 3043 passed、7 个精确 CUDA-only deselect、8 subtests passed，零 failure、skip 和 xfail，用时 202.62 秒。当前状态是等待独立 push 与 H100 recovery 授权；这不代表 RC4.3.7 已完成。
