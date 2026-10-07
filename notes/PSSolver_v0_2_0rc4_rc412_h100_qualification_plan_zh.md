# PSSolver v0.2.0rc4 RC4.1.2 Plane Nyquist 单次 H100 资格计划

## 1. 目标与边界

RC4.1.2 只关闭已知问题 B5：Plane free-slip Stokes 在偶数周期轴上的 Nyquist 一阶导数语义。它同时验证两件事：候选在 H100 上仍满足 full-complex / Hermitian-half 的数学一致性，以及修复没有使真实 `legacy_production` timestep 或显存超出冻结阈值。

冻结的运行时对象为：

- A：`0ce1e898d0b46f74e1032a3a7e93b46ea8667692`，即 RC4.1.0 reproducer 加 rc3 运行时；
- B：`d358ed66fe75da68a3e17fed402a956f21f1dd37`，即 RC4.1.1 Plane Nyquist repair。

资格工具可以位于 B 的后继提交中，但不得改动 B 的 runtime source。资格通过不自动授权 merge、默认值晋升、发布或长期模拟，也不修改 PSSolver-Control 和 `nematics3d`。

## 2. 为什么数值正确性与生产性能要分开验证

CUDA 正确性门禁直接运行五个制造解 case，覆盖两个周期轴、轴旋转以及 `full_complex` / `hermitian_half` 两种存储。每个 case 必须有限，速度与压力的跨存储 relative L2 都不得超过 `1e-12`。

生产性能门禁则使用真实 Plane `legacy_production` 路径和冻结的 R128/R320 初态。修复会有意改变受污染偶数网格的最终状态，所以 A/B 最终状态不要求相同；但同一角色的三个 trial 必须给出相同 final-state SHA，A/B 同网格必须从相同 initial-Q SHA 出发。这样既不会把修复后的正确差异误判成回退，也不会放弃确定性检查。

## 3. CPU 与输入门禁

登录节点完整 CPU suite 精确 deselect 七个 CUDA-only node ID，其中第七个是本阶段新增的 H100 full/half storage equivalence test。除此之外，只允许既有 optional `nematics3d` dependency skip。任何 failure、额外 skip、xfail 或额外 deselect 都必须停止。

本地实际结果为 `2781 passed, 7 deselected, 8 subtests passed`，0 failure、0 skip、0 xfail；运行时间 199.90 秒。

正式输入固定为：

- R128：`/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R128/Q_0.npy`，SHA-256 `28c70b72118c6d55b7646919ef2157959f40f52a1585753bb0732565cc6895c6`；
- R320：`/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R320/Q_0.npy`，SHA-256 `84a934a828794a3976b4af897f85d60c79e0c5f545b5f8265a9a4abed1e2cb4a`。

两者都必须是普通、非符号链接、有限的 float64 数组，shape 分别为 `[128,128,32,5]` 和 `[320,320,80,5]`。

## 4. 唯一 H100 Job

只允许提交一个正式 Job，不自动重试。Job 必须分配一张 NVIDIA H100 PCIe，只有一个可见 CUDA device，调用 token 使用 `cuda`，实际分配身份必须记录为 `cuda:0`，TF32 matmul 和 cuDNN 都必须关闭。

执行顺序如下：

1. 验证 Git、A/B 提交、clean detached worktree、冻结输入和 qualification-support 文件身份；
2. 在 H100 上真实运行七个 CUDA-only tests，要求 7 passed、0 failed、0 skipped、0 deselected；
3. 使用候选运行五个 CUDA storage-equivalence case，并原子写出、重新读取报告；
4. 分别从 A/B worktree 启动独立 profiler 进程，按 `A/B, B/A, A/B` 平衡顺序运行 R128 和 R320 各三组配对 trial，共 12 份 JSON；
5. 用版本化 analyzer 一次性裁决 profile 完整性、身份、有限性、transform calls、graph breaks、性能和显存；
6. 全部通过后才写 summary、provenance、checksum manifest 和 `COMPLETE`。

profile 固定使用 float64、`dt=0.005`、activity 18、`cubic_half`、truncated projected transforms、real-first、Hermitian-half、spectral H/stress space、compiled pointwise、Q-gradient reuse、禁用 spectral refresh、10 个 warmup steps、50 个计时 steps、seed 24。每步 transform calls 必须为 7/32，三个 compile 阶段 graph breaks 都必须为 0，runtime fallback 必须为 false。

## 5. 非回退阈值

每个网格独立裁决：

- candidate/baseline mean timestep ratio `<= 1.03`；
- median timestep ratio `<= 1.03`；
- 每个 paired trial ratio `<= 1.05`；
- peak allocated ratio `<= 1.03`；
- peak reserved ratio `<= 1.03`。

这里不采用“必须至少 2/3 更快”的方向性门禁，因为本修复的首要目标是消除错误模式，且微小测量噪声不应把性能等价误判为失败。

## 6. 结论语义

全部门禁通过时，唯一允许的成功分类是：

```text
PASS_RC4_1_2_PLANE_NYQUIST_SINGLE_H100_NON_REGRESSION
b5_complete = true
eligible_for_rc4_2_planning = true
```

任何 solver、CUDA、输入、身份、性能、显存或外部 helper 问题都必须 fail-fast。外部 helper failure 需要单独归因，但同样不得在没有新授权时提交第二个 Job。
