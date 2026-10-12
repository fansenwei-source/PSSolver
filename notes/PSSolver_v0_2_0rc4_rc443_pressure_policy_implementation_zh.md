# PSSolver v0.2.0rc4 RC4.4.3 Provider 压力策略实现记录

## 结论

RC4.4.3 已实现冻结的 Channel 压力数值策略核心：production primal、functional primal、functional transpose、standalone transpose qualification 和小网格 unrolled oracle 现在使用同一套 active-mode RMS、二次幂工作缩放、相对 breakdown 与停止后置条件。

本阶段分类为：

`PASS_RC4_4_3_PROVIDER_PRESSURE_POLICY_IMPLEMENTATION`

这不是 RC4.4 的最终关闭。公共声明、capability、runtime identity、checkpoint identity 与 restart 负向门禁仍属于 RC4.4.4 及后续资格阶段。

## 已实现的数值语义

- `fixed_iterations` 非空时派生为 `fixed_work`：固定工作量是权威要求，普通相对容差仅用于诊断，不能提前终止；仅 projected RHS 或 residual 精确为零时允许提前成功。
- `fixed_iterations` 为空时派生为 `relative_tolerance`：`max_iterations` 是权威上限，相对容差是必须满足的后置条件；到达上限但未满足容差时失败关闭。
- production 保留 persistent warm start，但只在求解成功后提交新 guess；失败时原 guess 保持对象和值不变。
- functional primal/transpose 始终从零开始，不读取也不写入 production warm start。
- operator 或 preconditioned-residual breakdown、非有限输入/中间量、以及 tolerance cap 未满足全部失败关闭，不发布部分结果。

## 尺度安全实现

- 绝对尺度使用 projected pressure active subspace 上的 complex-coefficient RMS。
- 分量尺度取活动系数 `real`/`imag` 绝对值的最大值。
- 工作向量按“不大于 component amax 的最大精确二次幂”缩放。
- breakdown 判据为：

  `abs(mean_active(conj(left) * right)) <= eps * active_rms(left) * active_rms(right)`

- 不再使用 `1e-30` 等绝对 breakdown 阈值。
- 已覆盖三个网格和 `2^-1074`、`2^-1020`、`2^0`、`2^1018`、`2^1023`。非零次正规输入不会被误判为零；不能用正 float64 表示的 RMS 使用 binary mantissa/exponent 和 log2 元数据保存。
- 普通尺度、相同迭代与终止签名下，functional primal/transpose 与未缩放旧运算次序逐字节一致。

## diagnostics schema v2 核心

每次求解生成不可变、JSON-safe 的完整快照，包含派生策略、活动自由度、RHS/初始/最终绝对残差、归一化残差、工作缩放指数、迭代数、终止原因与后置条件结果。

新的 primal 会立刻把公开 transpose 快照重置为同一 sequence 的 `not_run`。多步 autograd 仍可以逆序计算旧 primal 的 transpose，但旧结果不会覆盖当前 primal 的公开快照。

版本核心已更新为：

- pressure diagnostics schema：`2`
- pressure transpose protocol：`2`
- implicit pressure adjoint：`2`

这些版本尚未连接到公共 runtime/checkpoint identity；连接工作明确留给 RC4.4.4。

## 本地验证

- RC4.4.3 数值策略实现测试：28 passed。
- Channel 与 Phase 9.7 直接回归：211 passed。
- 受影响合同与历史 successor 定向测试：121 passed。
- 完整 CPU suite：3111 passed，7 deselected，0 failed，8 subtests passed。
- 七个 deselect 精确对应冻结 RC4 CPU 合同中的既有 CUDA-only node ID；本阶段没有提交 H100 作业。
- `git diff --check`：PASS。

## 失败类型

- production：`pssolver.linear_solvers.stokes.ChannelPressureConvergenceError`
- functional：`pssolver.functional.FunctionalConvergenceError`

两条路径都会在抛错前保存对应的完整失败快照。

## 明确未做

- 没有修改 PSSolver-Control。
- 没有修改 nematics3d。
- 没有合并四个 PCG loop；R5 重构仍在 v0.2 之后。
- 没有修改 public Simulation 用户输入的五字段形状。
- 没有修改生产默认值。
- 没有运行或提交 H100 作业。
- 没有 push。
- 没有宣称 B25、B28、B29 或 RC4.4 已经最终关闭。
