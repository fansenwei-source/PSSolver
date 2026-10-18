# PSSolver v0.2.0rc6 RC6-2：Channel 压力容差 floor 合同

RC6-2 处理独立 rc5 验收中的 S2-02 和 S2-04。本阶段不改压力方程、PCG 算法、checkpoint schema、生产默认值，也不在求解热路径增加一次压力算子计算。

## 决策

- tolerance mode 的最小公开支持值由 `2*finfo(real_dtype).eps` 提高为 `8*finfo(real_dtype).eps`。
- `relative_tolerance_met` 明确表示投影后的 **PCG recurrence residual** 满足请求值；不声称运行时另外重算了 returned-state residual。
- legacy Channel 与 complete-stress Channel 的公共编译器都会在 runtime 构造和输出目录创建前拒绝 sub-floor tolerance，公共错误码为 `discretization_contract`。
- exact floor 允许通过。
- fixed-work 模式中的 tolerance 仍只用于诊断，因此允许低于 floor。
- 既有 float32 默认 `1e-6` 高于新 floor，保持不变。

## 实现身份

- 分支：`fix/v0.2.0rc6`
- 合同实现提交：`d0690cce467d91f1b5fa5194af5f892a4678850b`
- 架构依赖修正及最终 qualified source：`a134658d04c9a6c0aa2cffd786982548e00f18f7`
- 本记录提交不是 qualified source。
- 未 push，未运行 H100。

## 验证边界

新增 CPU 回归覆盖 dtype/string dtype floor、float32/float64 returned-state residual 矩阵、两个公共 Channel 编译器的结构化拒绝、exact-floor 接受和 fixed-work 例外。旧 RC4/RC5/RC6-1 记录保持不变，通过本记录的 forward-only successor 表解析当前源码。

## 最终 CPU 结果

- RC6-2 直接合同：10 passed。
- 受影响行为集合：158 passed。
- 架构 import boundary：PASS；配置层不依赖线性求解器实现，纯 floor 规则位于 `pssolver.core.pressure`。
- 完整 suite：3871 passed，7 skipped，8 subtests passed，0 failed。
- skip 范围只有六项既有 CUDA-only 测试和一项可选 `nematics3d` 依赖测试。

最终分类：`PASS_RC6_2_CHANNEL_PRESSURE_FLOOR_CPU_CLOSURE`。RC6-2 完成，可进入 RC6-3 的本地规划，但没有自动 merge、default promotion、push 或 H100 授权。
