# PSSolver v0.2.0rc6 RC6-3 编译前置合同关闭记录

## 最终结论

`RC6-3 / S1-08 + A1-06` 已完成本地 CPU 关闭：

`PASS_RC6_3_COMPLETE_STRESS_COMPILE_PREFLIGHT_CPU_CLOSURE`

本阶段把 Periodic 和 complete-stress Channel 的必要模型参数，以及已经存在的 Q snapshot 的形状验证，移动到了公共 `compile_simulation()` 边界。错误现在以稳定的 `PublicSimulationCompilationError` 和结构化错误码报告，不再延迟到 runtime 泄漏原生 `KeyError` 或普通 `ValueError`。

## 实现

- 新增共享的 complete-stress 参数验证器，覆盖顶层 `material`、`ldg_l1`、`activity_amplitude`、`stokes`，以及 material/stokes 的必要嵌套字段。
- 缺失模型参数在编译期以 `malformed_application_spec` 拒绝。
- 对已经存在的 `.npy` Q snapshot 使用 memory-map 方式读取 header，不分配完整 payload；形状不匹配或 header 无法读取时，以 `initial_condition_contract` 拒绝。
- 接受 runtime 已支持的两种 Q 布局：`(*grid_shape, 5)` 和 `(5, *grid_shape)`。
- snapshot 尚未 staging 时仍允许纯规划编译；runtime 保留 dtype、finite 和 shape 的 TOCTOU backstop。
- 拒绝发生在 runtime 构造、timestep 和 output 创建之前。

实现提交：

`3070d852685326e7efa051bb76ae4ef6e3458962`

## 验证

- RC6-3 合同与架构边界：`38 passed`
- 受影响的 Periodic/Channel application、functional、checkpoint 与 RC6-2 回归：`145 passed`
- 历史 successor 审计第一组：`121 passed`
- 历史 successor 审计第二组：`122 passed`
- 完整 CPU suite：`3910 passed, 1 skipped, 8 subtests passed, 0 failed`
- 唯一 skip：既有 optional `nematics3d` dependency 不可用。
- `git diff --check`：PASS。

## 历史证据处理

既有 RC4、RC5、RC6-1 和 RC6-2 JSON 记录均未改写。所有被 RC6-3 合法修改的实现、CHANGELOG 和审计测试都通过新的 forward-only `historical_successor_sources` 登记解析。

## 边界

本阶段没有修改：

- 科学方程；
- integrator 或 transform 算法；
- checkpoint schema；
- runtime 或 production default；
- PSSolver-Control；
- nematics3d。

没有运行 H100，没有 push，也不授予自动 merge 或默认提升资格。

## 下一步

`eligible_for_rc6_4 = true`。下一阶段按冻结计划处理 `RC6-4 / A7-03`，在开始前另行核对精确问题合同。
