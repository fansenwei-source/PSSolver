# PSSolver v0.2.0rc3：PSSolver-Control 本构符号兼容记录

日期：2026-10-06

机器可读记录：
[`PSSolver_v0_2_0rc3_constitutive_compatibility.json`](PSSolver_v0_2_0rc3_constitutive_compatibility.json)

## 结论

从 PSSolver `d878abd3caecffb99f2cbf9255963a42f401af11` 到
`bb4489075301176c78772b07d14021432d4ad2b1`，PSSolver-Control planar
adapter 使用的六个 Beris--Edwards 本构 helper 的定义逐字节相同，
`q_tensor.py` 也逐字节相同。因此，Control 可以把 Provider commit/blob
pin 更新到 `bb44890`，随后运行自己的完整资格验证。

这是一份符号级兼容记录，不是整个 `beris_edwards.py` 的等价声明，也不是
PSSolver-Control 已经通过 rc3 资格验证的声明。

## 已验证的 Consumer 符号

以下六个 Provider 函数使用 AST `FunctionDef` 的原始源码区间提取，并分别对 UTF-8
字节计算 SHA-256。baseline 与 target 的哈希完全相同：

- `beris_edwards_active_stress_components`；
- `beris_edwards_algebraic_stress_components`；
- `beris_edwards_bulk_molecular_field_components`；
- `beris_edwards_distortion_stress_components`；
- `beris_edwards_molecular_field_components`；
- `beris_edwards_q_nonlinear_components`。

精确哈希记录在配套 JSON 中，并由 Provider 测试对当前 target 源码持续验证。

## 文件身份与差异边界

`pssolver/models/active_nematics/q_tensor.py` 在两个提交中的 Git blob 均为：

```text
5427df4402f31fc43ac409e1936cd31e58269966
```

`beris_edwards.py` 的 blob 从：

```text
681f8bdbc9cf4a6c06bbc8b86a044cfafb788217
```

变为：

```text
07f1f03f7f2f35e28e8ce629c1d73c5d59e4f5f3
```

该文件在此范围内唯一改变的顶层定义是 `BerisEdwardsQNonlinearModel`。提交
`5e1ef3ed6cf5954e723f10ae923de5873d010253` 为它增加了可选的
velocity-gradient inverse-transform 路径。这个 Provider runtime adapter 类不属于
本记录列出的六个 Control helper，不能用本记录宣称其自身与旧版逐字节等价。

## 对 Control 的授权边界

本记录允许 Control 维护者：

1. 在独立提交中把 planar adapter 的 Provider commit/blob provenance 更新到
   `bb4489075301176c78772b07d14021432d4ad2b1`；
2. 保留六个消费符号及其物理实现不变；
3. 更新 `docs/reference_pssolver.json`、兼容性文档和 provenance 测试；
4. 随后在未绕过 pin 的条件下重新运行完整 CPU suite。

本记录不允许：

- 删除或绕过 pin；
- 把 whole-file 差异描述为 whole-file 等价；
- 放宽 Consumer 测试；
- 宣称 Consumer 已完成 rc3 qualification；
- 修改生产默认值；
- 修改 nematics3d。

Control 的 pin 更新、Taylor validator 更新与 N2 Output policy 迁移应保持为三个独立
提交。只有完整 CPU suite 达到零 failure 后，才进入分支 push 和后续 installed-wheel/H100
累计资格验证。
