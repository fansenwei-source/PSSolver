# PSSolver RC4.1.2 Plane Nyquist H100 关闭记录

RC4.1 已完成，已知问题 B5 从 `partial_residual` 转为 `closed_verified`。权威分类为：

`PASS_RC4_1_2_PLANE_NYQUIST_H100_NON_REGRESSION_WITH_AUTHORIZED_HOST_COMPILER_RECOVERY_V5`

## 实现边界

通过资格验证的数值实现提交是 `d358ed66fe75da68a3e17fed402a956f21f1dd37`。它统一了 Plane free-slip Stokes 在两个偶数周期轴上的一阶导数 Nyquist 约定，并使 full-complex 与 Hermitian-half storage 得到一致的速度和压力结果。

后续 `ac9f5df` 与 `c39dabb` 只增加 qualification support，没有修改 `pssolver/` runtime。生产默认值、公共 API、checkpoint identity、lifting 和 pressure policy 均未改变；PSSolver-Control 与 `nematics3d` 未修改。

## 正确性证据

Recovery v4 提供并冻结了以下通过证据：

- 定向 CPU：76 passed；
- 完整 CPU：2789 passed、1 个 optional `nematics3d` skip、7 个授权 CUDA deselect；
- H100 CUDA-only tests：7 passed；
- 五组 Plane storage diagnostic 全部通过；
- full-complex 与 Hermitian-half 的最大速度或压力 relative L2 低于 `1e-12`。

v4 的整体 `FAIL_RC4_1_2_PROFILE_COMPILER_ENVIRONMENT` 分类保持不变；它的单个 partial profile 没有被复用。

## Host compiler recovery

Job `10859206` 在 `gpu-h100-4-0` 上使用 GCC/G++ 7.3.0，完成 `stdatomic.h`、CUDA driver shared library、fresh Triton `CudaUtils` 和 fresh CUDA `torch.compile` 编译门禁。TF32 保持关闭，没有 compile fallback 或 runtime fallback。

此前的 `/usr/bin/gcc` 4.8.5 环境问题属于外部工具链阻断，不是 solver 或科学失败。

## H100 非回退结果

新的 12/12 profiles 全部完成，每份均满足：

- 60 completed steps；
- finite；
- forward/inverse transforms per step = 7/32；
- graph breaks = 0；
- TF32 = false；
- 独立且初始为空的 Triton/TorchInductor cache。

R128 candidate/baseline mean、median ratio 分别为 `1.016821506` 和 `1.016249248`；R320 分别为 `1.009841762` 和 `1.009183051`。所有 paired trial 均低于 `1.05`，显存比均低于 `1.03`。

底层 analyzer 分类为：

`PASS_RC4_1_2_PLANE_NYQUIST_SINGLE_H100_NON_REGRESSION`

## 归档

最终控制目录：

`/home/fansenwei/pssolver_rc4_plane_nyquist_h100_c39dabb_20261007_recovery_v5`

`checksums.sha256` 包含 50 项，50/50 PASS；manifest 自身 SHA-256 为：

`0f891b57d7b665bae6a0d0dd29839fbf1955853f96a9cdccb29d3fab77514717`

`COMPLETE` 已生成。

本地收尾定向测试为 33 passed；按冻结合同排除 7 个 CUDA-only node 后，完整 CPU suite 为 2796 passed、7 deselected、8 subtests passed、0 failed。

## 下一步授权

RC4.1 和 B5 已关闭，只授权进入 `rc4.2_checkpoint_identity` 的规划阶段。目前不授权自动 merge、默认值提升、rc4 发布或 RC4.2 实现。
