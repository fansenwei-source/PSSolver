# PSSolver v0.2.0rc4 RC4.2.6 functional checkpoint 跨设备身份修复

## 结论

RC4.2.6 recovery v4 在 H100 上完成了 TF32 bootstrap 修复、7/7 CUDA-only 测试以及 X01–X10 production checkpoint 矩阵，但 X11 Periodic functional CPU→CUDA restore 在载入 tensor 前被 device-sensitive production runtime identity 拒绝。

这不是数值、科学、CUDA 或 qualification harness 失败。失败证据同时表明，去除 device/provenance 后的 forward metadata 完全相同；差异来自过滤前预计算并被继续保留的 `execution.sha256`。

## 修复

- functional forward identity 先递归去除 device、snapshot 和 run provenance，再重新计算 identity section digest；
- Periodic current functional-v3 restore 使用四层 checkpoint compatibility identity，而不再把 production runtime provenance SHA 当作跨设备兼容门禁；
- carrier 顶层 runtime SHA 仍必须与 bridge 中记录的 production provenance 一致；
- production checkpoint 和 legacy functional reader 的严格门禁保持不变；
- 已验证并密封的 pre-recovery current-v3 checkpoint 可以在重新规范化旧 execution digest 后继续读取；
- 未修改 solver timestep、模型方程、数值参数、公开 functional API 版本或生产默认值。

CPU 定向门禁为 93 passed；完整门禁为 2926 passed、7 个精确 CUDA-only deselect、8 subtests passed、0 failed、0 xfail。

## 当前边界

本提交只形成 CPU-qualified Provider 修复候选，不宣称 RC4.2.6 已完成，也不包含新的 H100 授权或结果。下一次 H100 recovery 必须重新执行 X01–X14 完整矩阵，且不得复用 v4 中未完成的 functional cells。
