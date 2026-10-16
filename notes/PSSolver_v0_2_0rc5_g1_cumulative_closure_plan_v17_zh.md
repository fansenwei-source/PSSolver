# PSSolver v0.2.0rc5 G-1 累计关闭计划 v17

## 结论

本记录冻结一个不需要新 H100 Job 的 CPU-only 累计关闭步骤。它不会重新解释或覆盖历史归档，而是把已经独立通过的证据按集合关系重新组合：

- G-1 CPU 与 installed-wheel CPU 关闭；
- rc1–rc4 checkpoint 矩阵与 provider identity 影响审计；
- recovery v5 已通过的 H100 环境、CUDA-only、TF32、checkpoint、数值与 Consumer 门禁；
- recovery v6 中已通过的 transform measurement-window 以及 12 个非 focused performance/memory cells；
- v16 中重新测量并通过的 4 个 focused balanced-crossover cells。

v6 的失败 cell 集合必须与 v16 的资格化 cell 集合精确相同。因此最终覆盖必须严格为 `12 + 4 = 16`，不得缺项、重叠或引入额外 cell。

## Fail-closed 规则

累计 analyzer 会重新校验五份 `checksums.sha256`，并重新散列 v6 使用的 96 份原始 profile。它还会拒绝以下情况：

- v5 或 v6 的历史失败状态被改写；
- v16 出现自动重试、第二个 Job、requeue 或 restart；
- 任意一个 focused cell 缺失或失败；
- 任意 archive manifest、输入 JSON 或 source profile 身份漂移；
- v6 修改了 profile 或阈值；
- 12 个复用 cell 与 4 个重新资格化 cell 不能精确覆盖全部 16 个 cell。

## 边界

本步骤不会提交 Slurm Job，不运行科学 timestep，不修改 solver/runtime、数值阈值、生产默认值、PSSolver-Control 或 nematics3d。v16 的旧 maximum-single-trial 估计器仍保留为 diagnostic-only，不参与新的正式裁决。

若累计 analyzer 通过，可以写入：

`PASS_RC5_G1_CUMULATIVE_CLOSURE_WITH_BALANCED_CROSSOVER_PERFORMANCE_QUALIFICATION`

并把 `g1_complete` 设为 `true`。这只授予 release integration review 资格；不会自动 merge、default promotion 或发布。
