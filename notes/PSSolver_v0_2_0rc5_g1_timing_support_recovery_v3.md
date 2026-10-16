# PSSolver rc5 G-1 timing instrumentation support recovery v3

状态：`READY_FOR_G1_TIMING_INSTRUMENTATION_SUPPORT_RECOVERY_V3`

Recovery v2 的唯一 H100 Job `10886063` 在 batch script 静态文件门禁处以 `FAILED / 1:0 / 00:00:00` 停止。缺失文件为：

```text
benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py
```

该失败发生在 node-local `/tmp` preflight、环境采集、GPU runtime、solver 构造和任何 profile 之前。它是 external qualification-support packaging failure，不是 solver、数值、性能、显存或 CUDA failure。Recovery v2 没有自动重试，归档 checksum manifest SHA-256 为 `1c28f842bb4a9b98e7a804f605ce989e6c859352d21eef6c487003b8a667c595`。

## 修正

此前 G-1 checkpoint recovery 已实际使用的 helper 已逐字节纳入仓库：

```text
benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py
SHA-256 ec6b4d2c70f5b4454754120b0f87a8771878ec50787a5590e23759bcc23cf142
```

timing instrumentation 的 plan-only preflight 现在冻结并逐项验证以下 Job support inventory：

1. checkpoint node-local `/tmp` preflight；
2. installed-wheel environment capture；
3. timing profile child helper。

缺少任一普通文件或出现 symlink 时都会在 `sbatch` 前 fail closed。原有 installed venv Python symlink 接受合同和 source-shadow 拒绝合同均保持不变。

## 验证边界

HPCC CPU 验证结果为：新测试 `10 passed`，新测试及相关门禁 `46 passed`，全部 RC5 G-1 合同测试 `168 passed`；均为 0 failed、0 skipped。installed-wheel plan-only preflight 覆盖 helper 身份、原子 JSON no-overwrite、缺失 helper negative gate、12 项矩阵及输出目录不创建。没有申请 GPU，也没有改变 production solver、runtime、checkpoint 实现、性能阈值或 v10 正式失败分类。

当前没有新的 push 或 H100 授权。后续必须先获得明确 push 授权和一次新的 analysis-only H100 recovery submission 授权；不得自动重试。
