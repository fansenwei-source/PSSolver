# PSSolver v0.2.0rc6 installed-wheel single-H100 release smoke

最终分类为 `PASS_RC6_INSTALLED_WHEEL_SINGLE_H100_RELEASE_SMOKE`。正式 package source 是 `fd7323d5e9ef87c537ed07476f8bd50b982af443`；发布准备证据 tip `dbb7f8f6f1d1132c71e82734ebefb74b55b7172c` 和本登记提交都不是 package source，也不要求重新运行 H100。

唯一正式作业为 Slurm Job `10901816`，在 `gpu-h100-4-0` 上以 `COMPLETED / 0:0` 结束，用时 `00:00:26`。本轮只提交一个 H100 Job，没有自动重试，也没有第二次提交。

Fresh installed-wheel 环境从冻结 wheel `pssolver-0.2.0rc6-py3-none-any.whl` 导入，wheel SHA-256 为 `9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5`，没有 source shadow import。环境为 Python 3.10.18、PyTorch 2.5.1+cu121、CUDA 12.1 和 NVIDIA H100 PCIe，functional API 仍为 1.0。

TF32 policy 在与 pytest 相同的 Python 进程中于测试前设置；pytest 前、pytest 后和 Plane one-step 后均为 `highest / false / false`。七个精确 CUDA-only tests 为 7 passed、0 failed、0 skipped、0 xfailed。

Installed-wheel Plane CUDA one-step 使用 `legacy_production` 和 eager pointwise execution，完成 1 步，未发生 runtime fallback，`COMPLETE` 存在，Q、速度和压力均有限。其 SHA-256 已冻结在机器可读记录中。

证据登记后的完整 CPU suite 为 4080 passed、7 deselected、8 subtests passed、0 failed、0 skipped、0 xfailed；七个 deselected node ID 正是已在本次 H100 Job 中真实运行并通过的 CUDA-only tests。

HPCC control 归档为 `/home/fansenwei/pssolver_v0_2_0rc6_release_h100_fd7323d_20261019_v1`，716/716 checksum PASS，manifest SHA-256 为 `f7e39759c6e1d62990d6eeeaf9fc18c8d68979889a17f1c5c0580996566bdda6`。scratch 归档为 `/scratch1/vincent/PSSolver/data_pssolver_v0_2_0rc6_release_h100_fd7323d_20261019_v1`，14/14 checksum PASS，manifest SHA-256 为 `1f9bd052266d794f9eef23785bc88ebb76c82301324702d5937b3e60c18acb2f`。两个目录均存在最终 `COMPLETE`。

本结果完成 RC6 的 installed-wheel H100 release smoke，并使该正式 package source 可以进入 annotated tag 与 GitHub prerelease 的单独评审。它没有授权或执行 tag、GitHub prerelease、PyPI 发布、自动合并或默认值提升；没有修改 PSSolver-Control、nematics3d 或生产默认值。
