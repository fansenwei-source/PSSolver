# PSSolver RC4.3.2：public lifting 能力与身份合同

## 结论

RC4.3.2 已完成合同冻结，分类为 `PASS_RC4_3_2_PUBLIC_LIFTING_CAPABILITY_RESTRICTION_CONTRACT`。本阶段没有修改求解器、checkpoint、public API 或生产默认值，也没有运行 H100。

RC4.3.1 已经证明两件事：当前 homogeneous-Dirichlet DST remainder representation 无法给出非零壁面分子场的收敛结果；lifted `molecular_field_linear_space = spectral` 与 `physical` 实际都执行 physical 路径。因此 RC4.3 不再把重新命名 metadata、warning 或近似执行当作修复，而选择 public fail-closed restriction。

## 精确限制边界

下列组合不再属于已资格化 public capability：

- model：`complete_stress_beris_edwards`；
- geometry：`plane_slab`；
- field：canonical five-component `Q`；
- boundary：两个 Plane 壁面上的 static constant prescribed Dirichlet；
- runtime：`legacy_production` 和 `compiled_v2`；
- molecular-field request：`physical` 或 `spectral`。

三种 public declaration `prescribed_q`、`strong_homeotropic_q` 和 `strong_planar_q` 继续可以导入和构造，但 capability catalog 必须标记为 `declarable = true`、`executable = false`，且 `qualified_applications = []`。这保留声明层和 field-neutral lifting 原语，明确区分“可以描述”与“已经科学资格化执行”。

以下能力保持不变：Plane 完整应力模型的 homogeneous-Neumann Q + free-slip velocity 路径、Periodic 完整应力、两条 Channel 路径，以及有限 Q Robin 的“可声明但未 public-qualified”状态。

## public compiler 的拒绝语义

RC4.3.3 应在 `pssolver.configuration.public_simulation_runner._compile_plane_public_simulation` 中、任何 run spec、lowering plan、construction plan、tensor、runtime 或输出目录产生之前拒绝该组合。

稳定合同为：

- exception：`PublicSimulationCompilationError`；
- enum：`UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD`；
- code：`unsupported_lifted_wall_molecular_field`；
- message token：`nonzero wall molecular-field semantics are not qualified for Plane static prescribed-Q lifting`；
- context 必须包含请求的 runtime/space、边界语义、受影响 field、支持的 `homogeneous_neumann` 子集以及 RC4.3.1 evidence digest。

`physical` 与 `spectral` lifted 请求必须得到同一错误码，不允许将 spectral 静默改成 physical。格式错误、只覆盖部分 Q 分量、非常数或非 Q 的 prescribed data 仍走已有的更严格 validation；合同不把它们变成受支持能力。

## requested/effective molecular-field space

任何成功的 public compile 都必须记录：

- `requested_molecular_field_linear_space`；
- `effective_molecular_field_linear_space`；
- `molecular_field_space_fallback_used`。

成功路径必须满足 requested == effective，且 fallback 为 false。非 lifted Plane 的 physical 请求执行并报告 physical；spectral 请求执行并报告 spectral。所有 lifted 请求在执行前拒绝。

## fresh initial conditioning

默认策略冻结为 `strict_no_automatic_conditioning_v1`：只减去 lift，不改变用户请求的 physical initial Q；若 fresh state 与边界不兼容则 fail closed。

可在 RC4.3.4 实现并资格化的显式 opt-in 是 `localized_quintic_wall_transition_v1`：

```text
x = clamp(distance_to_nearest_wall / transition_width, 0, 1)
smoothstep5(x) = 6*x^5 - 15*x^4 + 10*x^3
r_conditioned = smoothstep5(x) * r_raw
```

`transition_width` 使用 geometry 相同的物理长度单位，且满足 `0 < width <= Lz/2`。距离壁面大于等于 width 的区域必须保持 requested initial Q byte-for-byte 不变。

旧 whole-slab sine taper 被命名为 `legacy_whole_slab_sine_v1`，不得再成为隐式 public 默认，只能用于历史 provenance 和隔离的兼容性分析。canonical metadata 必须写明 policy schema、requested/effective policy、公式、width、单位、字段、分量和 `fresh_only`。

fresh conditioning 只属于 fresh-construction reproducibility，不进入 restart compatibility digest，也不允许重写 checkpoint 恢复状态。无 lifting 的初态必须与 RC4.2 baseline 逐字节一致。

## 版本与 checkpoint

RC4.3.3 只是 public capability restriction，不改变已经支持路径的数值动力学，因此不提升 checkpoint format、Plane forward-dynamics version、static-lifting plan schema 或 catalog schema；它只新增稳定的 public rejection code 和正确的 catalog 内容。

未来若真正修复 representation 并重新开放 lifted Plane，则必须提升 Plane forward-dynamics 与 lifting restart identity 版本，重新做 CPU/H100 qualification，并对旧 checkpoint 给出明确接受或 target mutation 前拒绝。旧 checkpoint 不原地重写，历史 reader 不因 public restriction 被删除。

## 下一步边界

下一步是 RC4.3.3：实现 compiler rejection、catalog 修正及成功路径的 requested/effective metadata，并运行 CPU negative/regression gates。RC4.3.4 才实现 fresh conditioning 策略。当前没有授权 push、H100、自动 merge、默认值提升或 release。
