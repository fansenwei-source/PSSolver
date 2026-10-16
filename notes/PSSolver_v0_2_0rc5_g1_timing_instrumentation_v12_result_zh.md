# PSSolver rc5 G-1 timing instrumentation v12 最终证据登记

最终分类：`ANALYSIS_ONLY_INSTRUMENTATION_EVIDENCE_COMPLETE`

归因分类：`NO_SINGLE_GC_REFRESH_OR_PRESSURE_TERMINATION_CAUSE_SUPPORTED`

唯一 H100 Job `10886072` 在 `gpu-h100-4-0` 上以 `COMPLETED / 0:0 / 00:02:39` 结束。12/12 个 candidate-only profile 均完成并通过结构、身份、有限性、TF32、telemetry 和双窗口状态一致性验证。归档 `checksums.sha256` 为 `83/83 PASS`，其自身 SHA-256 为：

```text
da7ddb92658572b76889123c56f08e7bdb6ef863b009409c42a1ab6379e58ee1
```

## 观测结果

- asynchronous 窗口共检测到 42 个 CUDA timing spikes；
- synchronized 窗口共检测到 34 个 spikes；
- 仅 1 个 spike 与 Python GC 重合，1 个与 scheduled refresh 重合；
- 180/180 条适用 pressure diagnostics 的 termination reason 均为 `fixed_work_completed`；
- synchronization 在 6 个 trial 中降低 CV，在另外 6 个 trial 中升高 CV；
- spike 位置随 trial 和窗口漂移。

所以当前证据不支持把 GC、scheduled refresh、pressure termination 或逐步 synchronization 中任一项认定为单一主因；它也没有识别出唯一替代主因。

## 严格边界

本记录不把 instrumentation 结果写成正式 qualification PASS。v10 的正式分类仍为 `FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE`。没有删除 timing sample，没有替换 trial，没有修改 1.03/1.05 性能阈值，没有修改求解器、runtime、科学方程或生产默认值。归档只有 `ANALYSIS_COMPLETE.json`，没有成功资格的 `COMPLETE`。

下一步是 CPU-only 预注册一种保留全部 60 个样本、保持六个 order-balanced pairs、并把稀疏位置漂移 spike 与候选回归区分开的 paired-block measurement contract。该设计和历史数据 replay 仍不能授予 G-1 qualification；正式采用合同和提交新 H100 Job 都需要后续明确授权。
