# PSSolver v0.2.0rc4 RC4.4.0 Channel pressure contract 方案

## 结论

RC4.4.0 已经把下一批工作冻结为一个边界清楚的 correctness batch：只处理 B25、B28 和 B29，不在 rc4 发布线上顺手重构全部 PCG，也不处理 RC4.5 的全局异常、文件权限或验证一致性问题。

本记录只完成规划。它没有改动 pressure 数值实现，没有修改 PSSolver-Control 或 nematics3d，没有提交 H100 作业，也没有把三个问题标记为关闭。下一步是独立的 RC4.4.1 analysis-only 裁决。

## 为什么不能直接改一个阈值

目前 Channel 有四份具有不同职责的 PCG 循环：

1. production primal，读取并写回 persistent pressure warm start；
2. operator qualification transpose，从零初值开始；
3. functional implicit adjoint 的 primal/transpose，共用从零初值开始的 graph-free 循环；
4. 小网格 CPU unrolled gradient oracle。

它们现在不是同一个停止合同：production 和 transpose qualification 使用绝对 `1e-30` breakdown 阈值，functional implicit adjoint 使用相对尺度阈值；只有 functional 路径会在极端 RHS 上归一化；所有路径都把 `fixed_iterations` 描述成确定性工作量，却仍允许 tolerance 提前结束；production 在达到上限或 breakdown 后没有强制成功后置条件。

因此只改 `1e-30`，或只在 adjoint 中多除一个范数，都不能关闭 B25/B29。

## 冻结的两种 pressure policy

公共构造参数在 v0.2 期间保持兼容：仍然是 `relative_tolerance`、`max_iterations` 和可空的 `fixed_iterations`。实现必须从它们派生唯一 mode，并把 mode、字段角色和后置条件写入 canonical metadata。

### fixed_work

当 `fixed_iterations` 非空时：

- `fixed_iterations` 是唯一权威工作上限；
- `max_iterations` 只是兼容字段，在此 mode 中不参与停止；
- `relative_tolerance` 只用于 diagnostics，不再作为普通 early exit；
- 除 projected RHS 或 residual 已经精确为零外，应完成指定工作量；
- 任何数值 breakdown 都 fail closed，不能把“提前 break”当成成功。

这才使 fixed mode 真正具有可重放、可比较的工作量语义。

### relative_tolerance

当 `fixed_iterations` 为空时：

- `max_iterations` 是权威上限；
- `relative_tolerance` 是必须满足的成功后置条件；
- 达到上限仍未满足 tolerance、出现 breakdown 或 non-finite 都必须 fail closed。

production 仍保留 warm start；functional primal 和 transpose 仍保持每次 zero start。RC4.4 不把二者伪装成同一种初值策略。

## B25 为什么先做 analysis-only

当前证据还不足以在 global L2、active-mode RMS 和 normwise backward error 之间直接选一个。RC4.4.1 必须在不改源码的条件下比较这些候选，并覆盖：

- float64 安全范围内从 subnormal-adjacent 到 quadratic-overflow-adjacent 的二次幂尺度；
- 至少三个不同 active pressure DOF 的 Channel 网格；
- primal、显式 conjugate transpose、production warm start 和 functional zero start；
- near-breakdown、zero RHS、小网格 dense solve 和 unrolled gradient oracle。

候选必须同时满足尺度协变、网格不敏感、有限性、primal/transpose 一致性和梯度正确性。正式结论出来前，不能把任一公式写成既定答案，也不能保留绝对 `1e-30` 作为权威门限。

## diagnostics schema v2

新的 diagnostics 必须让一次 solve 可以独立审计，而不是只留下 iteration/residual 三个标量。至少记录：

- solver/operator/phase/initial-guess identity；
- mode、所有请求参数、每个参数的角色和权威 iteration limit；
- active pressure DOF、RHS norm 定义和值；
- initial/final absolute residual、normalized residual 定义和值；
- iteration count、termination reason、required postcondition、postcondition satisfied 和 acceptable。

primal 与 transpose 的结果必须按 phase 隔离；新的 primal 不能让上一次 transpose 看起来像本轮已经运行。production diagnostics 也不能继续被标成 zero-start。

## B28 的实际情况

Phase 9 之后，顶层 declaration 和 runtime 已经声明 executable，custom adjoint 也确实已连接；所以不能照旧审计文字盲目改成另一套能力。

但当前 component metadata 仍有真实歧义：implicit-adjoint component 写着 `functional_runtime_executable=false`，而它已经被 executable Channel functional runtime 使用。与此同时，独立的 transpose operator 本身确实不是一个完整 runtime。RC4.4 要把“component 自身是不是 runtime”和“component 是否已接入 executable runtime”拆开表达。

`explicit_vjp=false` 暂时保留，因为它表示没有公共显式 `vjp()` 方法，不等于没有 custom autograd adjoint。B28 只有在 source 与 installed-wheel 的 declaration/runtime/capability truth table 全部一致后才能关闭。

## identity 与 restart

fixed-work 停止语义、尺度处理和 adjoint contract 都会改变 dynamics identity。计划冻结以下目标版本：

- pressure convergence schema v2；
- pressure transpose protocol v2；
- implicit pressure adjoint v2；
- production forward `channel.production.forward.v4`；
- functional forward `channel.functional.forward.v4`；
- functional derivative `channel.functional.derivative.v4`。

state layout 与 backend layout 不变。旧 pressure dynamics checkpoint 不能静默当作新版本 exact restart，必须在 tensor load 和 target mutation 之前拒绝。若以后需要 state-only migration，必须另行授权和资格验证。同一新版本 continuous/split/file restart 仍要求 exact。

## 分阶段迁移

1. **RC4.4.1**：analysis-only 尺度与 capability 裁决；
2. **RC4.4.2**：冻结 policy schema v2、diagnostics、异常后置条件、identity bump 和正式门限；
3. **RC4.4.3**：实现 Provider pressure policy，但不合并四份循环；
4. **RC4.4.4**：连接 declaration、diagnostics、checkpoint 和 negative gates；
5. **RC4.4.5**：Provider source + installed-wheel CPU closure；
6. **RC4.4.6**：独立 PSSolver-Control migration + installed-wheel CPU closure；
7. **RC4.4.7**：一次 H100 累计关闭。

PSSolver-Control 的已资格 source 是 `03dbc39975978d9d0b7f84965015da24f9d3c206`，证据登记 commit 是 `c4b50e63cafba14d1985fbe4a71aa78212eaf050`。Consumer 必须等 Provider 发布最终 policy/identity record 后再改，RC4.4.0 本身不授权修改 Consumer。

## H100 关闭边界

RC4.4.7 需要单独明确授权，只允许一次正式 submission，不自动重试。它将从 installed wheels 运行 fixed-work 与 tolerance 两种模式、极端尺度子集、custom adjoint/FD/Taylor、checkpointed/full-history gradient、same-version restart、两组生产尺寸的性能和显存门禁。正式结果出来后不得修改阈值。

## 明确不做的事

- 不执行 R5 的统一 PCG 框架重构；
- 不修改 Plane 或 Periodic pressure；
- 不添加几何、模型或边界条件；
- 不修改 production runtime 的默认选择；
- 不修改 PSSolver-Control 或 nematics3d；
- 不授权 optimizer campaign、larger batch、自动 merge、默认提升或发布。

下一步应当是 RC4.4.1，而不是直接进入实现。
