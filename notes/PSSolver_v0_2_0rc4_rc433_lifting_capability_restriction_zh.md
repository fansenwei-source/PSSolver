# PSSolver RC4.3.3：public lifting 能力限制实现

## 结论

RC4.3.3 已完成，分类为 `PASS_RC4_3_3_PUBLIC_LIFTING_CAPABILITY_RESTRICTION`。本阶段把 RC4.3.2 冻结的合同落实为 public compiler 的 fail-closed 行为，没有修改底层 lifting 数学、已支持路径的数值动力学、checkpoint schema 或生产默认值。

## 公开能力的精确边界

Plane complete-stress Beris–Edwards 的 static two-sided prescribed-Q lifting 不再被声明为已资格化 public execution。`prescribed_q`、`strong_homeotropic_q` 和 `strong_planar_q` 仍可导入和构造，但 capability catalog 现在如实报告 `executable = false` 且没有 qualified application。

public compiler 对以下 12 种组合给出相同的稳定拒绝：三种 Q boundary declaration × `legacy_production`/`compiled_v2` × molecular-field `physical`/`spectral`。错误码为 `unsupported_lifted_wall_molecular_field`，context 记录 model、geometry、field、boundary 语义、requested runtime、requested molecular-field space、仍受支持的 `homogeneous_neumann` 子集和 RC4.3.1 evidence digest。

拒绝发生在 run spec、lowering plan、construction plan、tensor、runtime 和输出目录产生之前。格式错误或只覆盖部分 wall face 的声明仍由原有 boundary validation 拒绝，并未因此成为受支持能力。

## 保留的行为

Plane homogeneous-Neumann Q 路径继续支持 `legacy_production` 与 `compiled_v2`。Periodic complete-stress 和 Channel complete-stress 也保持原连接。三个成功编译器现在统一记录：

- `requested_molecular_field_linear_space`；
- `effective_molecular_field_linear_space`；
- `molecular_field_space_fallback_used = false`。

成功路径强制 requested == effective，不允许把 lifted spectral request 静默改成 physical。

底层 field-neutral lifting primitive、internal lowering、legacy internal binding 和历史 checkpoint reader 均未删除。这些保留项用于历史证据、兼容读取和未来真正修复 representation，但不再构成 public scientific qualification 声明。

## CPU 验证

定向合同集合为 31 passed；完整相关 public/lifting 回归集合为 125 passed。覆盖了 12 种受限组合、partial declaration 的既有严格 validation、preconstruction 顺序、catalog 内容、稳定错误 metadata、homogeneous-Neumann 成功路径、Periodic/Channel requested/effective metadata，以及低层 lifting 能力仍可用。

本阶段不改变已支持数值路径，因此不需要 H100。没有运行 H100，也没有修改 PSSolver-Control 或 `nematics3d`。

## 下一步边界

可以进入 RC4.3.4：把 fresh lifted initial-state conditioning 从隐式 whole-slab sine taper 改为冻结合同中的显式策略。RC4.3.4 尚未授权；本阶段也不授权 push、merge、默认值提升或 release。
