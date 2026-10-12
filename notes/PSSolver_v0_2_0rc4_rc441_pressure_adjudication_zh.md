# PSSolver RC4.4.1 Channel pressure 尺度与 capability 裁决

## 结论

RC4.4.1 已完成，分类为：

`PASS_RC4_4_1_ACTIVE_MODE_RMS_AND_SCOPED_CAPABILITY_CONTRACT`

本阶段只新增可重复运行的分析工具、裁决记录与回归测试，没有修改 pressure solver、生产默认值、PSSolver-Control 或 nematics3d。RC4.4 尚未完成；下一阶段是 RC4.4.2 合同冻结，不是直接修改求解器。

## B25：最终选择的尺度定义

公开的 RHS 和 residual 绝对尺度采用 active pressure modes 上的 overflow-safe RMS：

`amax * ||active_values / amax||₂ / sqrt(active_pressure_degrees_of_freedom)`

归一化残差定义为：

`active_mode_rms(residual) / active_mode_rms(projected_rhs)`

PCG 内部工作向量另用 active-mode `amax` 推导出的精确 2 的幂缩放。这样既不让任意十进制除法改变不必要的运算，也能确保内积和二次量在安全范围内。breakdown 使用相对于两个参与向量尺度的 machine-epsilon 判据，不再允许 `1e-30` 一类绝对 magic threshold。

为什么没有选择一个 float64 global L2 作为公开绝对尺度：对于每个系数都有限的合法向量，global L2 仍可能因为 active degrees of freedom 而超过 float64 上限。三个测试网格在每个 active coefficient 都等于 `2^1023` 时，global L2 的数学 `log2` 分别为 1024.73、1025.04 和 1025.94，无法装入一个有限 float64；active-mode RMS 仍精确为 `2^1023`。在固定 active subspace 上，两者的相对残差比数学上相同，但 RMS 的公开绝对值还具有跨网格可比较性。

normwise backward error 仍保留为小网格独立资格诊断，不放进生产 hot path：它需要算子范数，成本和实现耦合都不适合作为每次 PCG 的停止条件。

## 数值矩阵

分析覆盖：

- 网格 `(3,2,2)`、`(4,3,2)`、`(5,4,3)`，active pressure degrees of freedom 分别为 11、17、59；
- primal 与显式 conjugate-transpose；
- fixed-work 与 relative-tolerance 两种模式；
- RHS 精确二进制尺度 `2^-900`、`2^-600`、`1`、`2^600`、`2^900`；
- 额外 norm stress：`2^-1074` 到 `2^1023`。

全部求解结果有限且可接受。相对于 scale=1 结果的最大 pressure scale-covariance relative L2 为 0。fixed-work 在所有网格与尺度上精确执行 5 次并以 `fixed_work_completed` 结束；tolerance 模式的迭代数分别稳定为 5、11、19。

小网格独立 dense oracle 结果：primal 与 transpose 的 PCG/dense relative L2 均为 `1.8664e-16`；normwise backward error 均为 `5.5370e-17`；primal/transpose 内积误差为 `4.6907e-16`。custom VJP 相对 unrolled oracle 的 relative L2 为 `8.9223e-17`，中心差分误差为 `2.7215e-11`，Taylor 阶数约为 2。

production warm-start 与 functional zero-start 语义保持分离：同一 RHS 的 production 首次求解为 11 次，warm-start 再求解为 0 次；functional 求解没有覆盖 production pressure guess，且结果与首次 production 求解逐位一致。

独立诊断连续运行两次，完整 JSON 的 SHA-256 都是 `1c3803295a582e9e92704acd45b922bea417c0ab0b2c239be408b9453dbe809a`。pressure/functional 定向回归为 95 passed；最终完整 CPU suite 为 3066 passed、7 deselected、8 subtests passed、0 failed、0 skipped。七个 deselect 仍是既有 CUDA-only node ID；本地环境可以导入 optional nematics3d，因此本轮没有 optional-dependency skip。

## B29：停止策略

RC4.4.0 的两模式合同得到数值支持并保持不变：

- `fixed_work`：`fixed_iterations` 是唯一权威预算，`relative_tolerance` 只用于诊断；除 projected RHS 或 residual 精确为零外不得普通提前退出；任意数值 breakdown 必须 fail closed。
- `relative_tolerance`：`max_iterations` 是上限，tolerance 是必须满足的后置条件；到达上限但没有满足 tolerance 必须 fail closed。

`fixed_iterations` 非空时，兼容保留的 `max_iterations` 是 inactive field。模式必须在配置阶段推导一次，之后配置 metadata、runtime、diagnostics 和 checkpoint identity 必须一致。

## B28：capability truth table

声明与实际 runtime 的 capability metadata 当前一致；两者都是真正 executable，并且使用 custom implicit pressure adjoint。`explicit_vjp=false` 仍然正确，因为公共 API 没有单独的显式 VJP 方法，这不等于“没有 custom autograd adjoint”。

真正需要修正的是组件级字段。`ChannelImplicitPressureAdjoint` 和 standalone transpose operator 都不是完整 runtime，因此 `component_is_runtime=false`；但两者都连接进 executable Channel functional runtime。当前共同使用的 `functional_runtime_executable=false` 把“组件本身不是 runtime”和“组件没有连接到 executable runtime”混在一起。

RC4.4.4 必须将它拆成至少：

- `component_is_runtime`
- `component_connected_to_executable_runtime`

standalone transpose 仍可如实保持非 runtime；implicit adjoint 则需如实记录它已经被 executable runtime 使用。

## 下一步边界

RC4.4.2 应冻结 schema v2 的每个字段含义、异常与 fail-closed 后置条件、identity 版本变化、CPU/H100 阈值，以及 source/installed-wheel 的精确门禁。此时仍不应修改 production solver。Provider 数值实现属于 RC4.4.3，capability/diagnostics/checkpoint 连接属于 RC4.4.4；PSSolver-Control 修改和 H100 submission 仍需分别授权。
