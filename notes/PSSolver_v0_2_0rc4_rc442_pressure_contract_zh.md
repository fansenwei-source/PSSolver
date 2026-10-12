# PSSolver RC4.4.2 公共 Channel pressure 合同

## 状态

RC4.4.2 已完成，分类为：

`PASS_RC4_4_2_PUBLIC_CHANNEL_PRESSURE_CONTRACT`

这一阶段只冻结合同，不修改求解器、functional runtime、checkpoint reader/writer、PSSolver-Control、nematics3d 或生产默认值。B25、B28、B29 仍是 `partial_residual`，要等 RC4.4.3–RC4.4.7 的实现和资格关闭后才能转为 closed。

## 公共配置保持兼容

用户仍然只提交五个既有字段：

- `algorithm`
- `relative_tolerance`
- `max_iterations`
- `fixed_iterations`
- `warm_start`

不增加用户必须填写的 `mode`。模式由 `fixed_iterations` 是否为空唯一推导：

- 非空：`fixed_work`，`fixed_iterations` 是权威预算；`max_iterations` 是 inactive compatibility field；tolerance 只用于诊断。
- 为空：`relative_tolerance`，`max_iterations` 是权威上限；tolerance 是必须满足的后置条件。

派生后的模式和字段角色必须出现在 declaration、compiler normalization、production/runtime identity、functional declaration/runtime identity 和 diagnostics 中。用户不能伪造这些派生字段。

## active-mode RMS 的无下溢表示

RC4.4.1 已选 active-mode RMS。RC4.4.2 进一步解决一个边界：合法非零 RMS 可能小于 float64 最小非零数，因此不能只保存裸 float。

schema v2 使用 `channel_pressure_active_rms_magnitude.v1`：

- `is_zero`
- `float64_value`
- `representable_as_positive_float64`
- `log2_value`
- `binary_mantissa`
- `binary_exponent`

零由 active coefficients 的精确零测试决定，不由 `float64_value == 0` 推断。非零但不能以正 float64 表示时，`float64_value=null`，其 log2 和二进制 mantissa/exponent 仍完整存在。JSON 中不允许 NaN/Inf。

对 complex coefficient，内部缩放基准是 active modes 上 `abs(real)` 与 `abs(imag)` 的最大值；这样即使实部和虚部都接近 float64 上限，也不会先在 complex magnitude 中溢出。PCG 工作尺度是这个最大分量以下最大的精确 2 的幂。

breakdown 使用 active-mode mean inner product 的相对 machine-epsilon 判据。`1e-30` 一类绝对阈值不再具有任何权威性。

## diagnostics schema v2

每份记录都必须包含求解序号、phase、initial-guess policy、两种模式的全部请求值和字段角色、active DOF、三个 scale-safe magnitude、归一化残差、工作缩放、breakdown 定义、迭代数、结束原因、后置条件和最终 acceptable。

fixed-work 只有以下情况可以成功：

- projected RHS 精确为零；
- 迭代中 residual 精确为零；
- 精确完成请求的 fixed iterations。

tolerance 模式只有精确零或实际满足 tolerance 才能成功。到达 cap、breakdown、非有限输入或中间量全部失败。

production pressure guess 只在成功后提交；失败不得替换旧 guess，也不得发布部分结果。functional 始终 zero-start，不增加 hidden pressure state。

implicit adjoint 的 aggregate diagnostics 带 `solve_sequence_id`。每次新 primal 都必须原子地把 transpose 重置为同一序号的 `not_run`；随后 backward transpose 必须指向当前 primal 序号，不能继续暴露旧 transpose 结果。这直接避免此前 qualification 中发生过的 diagnostics phase 污染。

## capability 语义

declaration 和 runtime 继续声明 executable、`validated_custom_adjoint`，且 capability set 必须相同。`explicit_vjp=false` 保持不变，它只表示没有独立公共 VJP 方法。

组件 metadata 移除模糊的 `functional_runtime_executable`，改为：

- `component_is_runtime`
- `component_connected_to_executable_runtime`
- `connected_runtime_kind`
- `custom_autograd_rule`
- `public_explicit_vjp_method`

implicit adjoint 和 standalone transpose 都不是完整 runtime，但都连接在 `channel_activity_batch_one` runtime 中；只有 implicit adjoint 拥有 custom autograd rule。

## identity 与 restart

压力策略改变实际 forward 和 derivative 数学路径，因此版本必须升级：

- convergence schema：1 → 2
- transpose protocol：1 → 2
- implicit adjoint：1 → 2
- production forward：`channel.production.forward.v3` → `v4`
- functional forward：`channel.functional.forward.v3` → `v4`
- functional derivative：`channel.functional.derivative.v3` → `v4`

state layout、backend restart layout、functional bridge format 和 checkpoint envelope 不变。相同 v4 continuous/split/file restart 必须逐字节一致；v3↔v4 必须在 tensor load 和 target mutation 前拒绝。没有隐式 state-only migration，也不重写历史 checkpoint。

## 后续资格门禁

Provider CPU 必须覆盖配置真值表、zero RHS、精确 fixed budget、tolerance cap fail-closed、三网格极端尺度、primal/transpose、breakdown 正负控制、dense oracle、custom adjoint、FD/Taylor、diagnostics phase isolation 和 restart negative gates。

预注册阈值包括：scale covariance `2e-11`、primal/transpose inner-product `2e-13`、PCG/dense `5e-12`、backward error `5e-14`、VJP/unrolled `3e-13`、central FD `2e-7`、Taylor order 最低 `1.95`。

最后的单次 H100 仍需另行授权。它必须使用 installed wheels、TF32 off、无 fallback/graph break，覆盖 fixed/tolerance、primal/transpose、scale、gradient、checkpoint 与两个 Channel profile 网格；mean/median timestep 和 peak allocated ratio 上限均冻结为 1.05。

## 本地验证

pressure、functional、transpose、implicit adjoint、runtime diagnostics、RC4.4.0/4.1/4.2 合同和总问题矩阵的定向门禁为 `122 passed`。排除冻结的 7 个 CUDA-only node ID 后，完整 CPU suite 为 `3078 passed, 7 deselected, 8 subtests passed`；没有 failure、skip 或 xfail。本地环境可以导入 optional nematics3d，本轮未修改 nematics3d。

## 下一步

下一步是 RC4.4.3：在不做 R5 全面 PCG 合并的前提下，分别实现 production primal、functional primal/transpose、standalone transpose 和小网格 oracle 的统一策略。RC4.4.3 尚未授权；PSSolver-Control 修改、H100、push、merge、默认值切换和发布也均未授权。
