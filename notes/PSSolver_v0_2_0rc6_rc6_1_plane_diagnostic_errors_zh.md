# PSSolver v0.2.0rc6 RC6-1：Plane compiled_v2 diagnostics 公共异常统一

## 结论

RC6-1 已完成本地 CPU 关闭：Plane `compiled_v2` 的 normal-force 与 pressure
diagnostics 一旦出现非有限值，统一抛出公开的 `SimulationStateError`。该异常同时属于
`SimulationExecutionError`，机器码为 `simulation_state_error`。

这关闭了 rc5 独立验收中的 `A4-14 / A7-02`。修复只改变错误类型及其结构化 metadata，
没有改变方程、积分器、变换、压力算法、checkpoint schema 或生产默认值。

## 公共错误 metadata

错误固定携带：

- `application_family = plane_complete_stress`；
- `stage = diagnostic`；
- `step = compiled workflow completed_steps`；
- `details.nonfinite_fields` 精确列出非有限的 diagnostic 字段。

normal-force 路径登记 `projected_normal_force`；pressure 路径分别检查迭代数、绝对残差和
相对残差。diagnostics 被禁用或尚不可用仍然是能力/调用顺序错误，继续抛出原有
`RuntimeError`，不会被错误地归类为状态非有限。

## 验证

- normal-force 非有限直接回归：PASS；
- pressure 多字段非有限直接回归：PASS；
- 公共继承关系、machine code、family、stage、step 和 details：PASS；
- diagnostics disabled 的既有行为：PASS；
- 直接与既有相关测试合计：`29 passed, 0 failed`；
- 扩展 compiled/记录定向门禁：`108 passed, 0 failed`；
- 完整 forward-only successor 审计集合：`195 passed, 0 failed`；
- 完整源码 CPU suite：`3857 passed, 7 skipped, 8 subtests passed, 0 failed`；
- RC4/RC5 冻结记录全部通过 forward-only successor 解析，旧记录未改写。

本阶段不需要 H100，没有修改 PSSolver-Control 或 nematics3d，也没有 push。

## 下一步

RC6-1 完成后可进入 RC6-2：先分析 pressure tolerance floor 的可实现边界，再修正
truthful postcondition 与 sub-floor compilation gate。RC6 的 source/installed-wheel 全量门禁和
H100 累计资格将在后续 source freeze 后统一执行。
