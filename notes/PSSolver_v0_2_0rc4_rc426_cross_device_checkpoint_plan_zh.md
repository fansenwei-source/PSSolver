# PSSolver RC4.2.6 单次 H100 跨设备 checkpoint 可移植性计划

## 目标

RC4.2.6 只回答一个问题：在科学、离散、runtime path、state layout、backend restart、lifting 和 derivative identity 全部相同的情况下，当前格式 checkpoint 能否在 CPU 创建后恢复到已分配的 H100，及在 H100 创建后恢复到 CPU。

设备 token 属于运行 provenance，不属于兼容身份。该结论不允许扩展成跨 runtime、跨 dtype、跨 shape、跨旧版本的任意 restart，也不声称 CPU 与 CUDA 在恢复后的浮点轨迹逐字节一致、性能相同或长期科学行为已经验证。

## 冻结矩阵

矩阵包含 14 个有序 cell，两个方向都必须覆盖：

- Plane `legacy_production`、`compiled_v2` 和 `separated_canary`；
- Periodic production `periodic_spectral`；
- Channel production `channel_complete_stress`；
- Periodic functional `periodic_activity_batch_one`；
- Channel functional `channel_activity_batch_one`。

每个 cell 使用 float64、batch one 和确定性解析初态。source 先推进一步再写 checkpoint；target 恢复后必须满足 compatibility identity、进度计数、所有序列化 tensor 和 persistent backend state 的精确相等，并再推进一步证明恢复结果可用且有限。后一步只检查成功和有限，不要求跨设备字节一致。

## 门禁与执行顺序

提交前必须验证 RC4.2.5 的 `COMPLETE` 和 26/26 checksum、Git 身份、干净 detached worktree、installed wheel、`pip check`、helper synthetic tests、X01–X14 唯一命令矩阵和空 scratch。

唯一 H100 Job 先持久化 decomposed 环境观测，再执行 enforcement；TF32 两个 Python flag 必须显式关闭。若触发 `torch.compile`，必须先验证 host compiler，并为 Triton/TorchInductor 使用隔离空 cache。随后真实运行既有七个 CUDA-only tests，再按 X01–X14 顺序运行；每个 cell 完成后原子写入报告。任何失败立即停止，不得自动重试或提交第二个正式 Job。

有效 scheduler 配置冻结为 `hagan-gpu`、`hagan-lab`、`medium`、`gpu:H100:1`、8 CPU、64G、1 小时和 no-requeue。

## 成功与范围

只有 14/14 cell 和所有前置门禁全部通过，才能给出：

`PASS_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY`

并将 RC4.2 标记完成、允许规划 RC4.3。即使通过，也不自动授权 merge、默认值提升或发布。

本提交只冻结计划。它不修改 checkpoint reader/writer、runtime、PSSolver-Control、`nematics3d` 或生产默认值，也不提交 H100。下一提交才可以实现 versioned helper、analyzer、synthetic tests 和 Slurm harness；真正提交 H100 仍需单独明确授权。

本地计划门禁结果为：19 项定向合同测试通过；完整 CPU suite 为 `2890 passed, 7 deselected, 8 subtests passed`，0 failure、0 skip、0 xfail，运行时间 204.22 秒。
