# PSSolver v0.2.0rc4 RC4.3.1 lifting 正确性裁决

## 结论

`classification = PUBLIC_CAPABILITY_RESTRICTION_REQUIRED`

当前的 Plane complete-stress Beris–Edwards static Q lifting 不能继续被描述为已经解决了非零 wall molecular-field 语义。RC4.3.1 没有修改求解器，而是用真实 `legacy_production` runtime、解析 Q 导数和独立物理空间参考完成了 L01–L05；所有数组有限，runtime fallback 均为 `false`。

机器可读完整记录：

`notes/PSSolver_v0_2_0rc4_rc431_lifting_correctness_analysis.json`

该文件 SHA-256：

`1157b08b5328996876307fb0333aa28e85932d9d09e9ed0f16f08cc7a3a62ef1`

## L01：标量 affine-lift control

wall-normal 分辨率：`16, 32, 64, 128, 256`。

RMS error：

`5.5981586491e-05, 1.4008891513e-05, 3.5030668534e-06, 8.7581947026e-07, 2.1895816509e-07`

相邻收敛阶：

`1.99861, 1.99965, 1.99991, 1.99998`

回归收敛阶：`1.99959`。这项 control 通过，说明诊断本身能够解析预期的二阶收敛。

## L02：真实 solver 的非零 wall molecular field

使用解析 in-plane director twist，把精确物理 Q 注入真实 lifted runtime；独立参考直接用解析 Laplacian 和本构关系计算，不调用被测 lifting projector。

wall molecular-field Linf error：

`6.33038e-04, 5.97049e-04, 5.80256e-04, 5.72109e-04, 5.68092e-04`

wall-error 回归阶只有 `0.03739`。bulk relative-L2 回归阶为 `0.53358`。物理 Q 本身在 wall cell 和 first-interior cell 与解析输入精确一致，因此失败来自 molecular-field representation，而不是 Q 注入误差。

每次真实 static refresh 的 transform calls 为 forward `6`、inverse `23`，没有 runtime fallback。

## L03：twisted Q 到 Stokes flow

使用解析 x–z director twist。参考路径直接计算完整本构 stress，再用独立的 dense physical-space quadrature 求 free-slip tangential velocity；pressure 参考使用完整 normal stress 并去均值。

velocity relative-L2 error：

`1.13991e-01, 5.33292e-02, 2.58348e-02, 1.26862e-02, 6.26428e-03`

回归阶为 `1.04429`，约为一阶而不是冻结合同要求的二阶。demeaned pressure relative-L2 回归阶为 `0.91689`。

## L04：fresh initialization conditioning

真实 runtime 对 uniform Q 和 production-style analytic periodic defect gas 均应用全域 `sin(pi*(j+1/2)/Nz)` remainder envelope。

- 每个分辨率上，exactly unchanged wall-normal cell 数均为 `0`；
- uniform Q 在 quarter height 的 Linf deviation 约为 `0.056–0.065`；
- defect gas 在 quarter height 的 Linf deviation 约为 `0.169–0.201`；
- canonical metadata 只有 `kind = sine_wall_taper` 与 wall values；
- metadata 没有公式、transition width 或 width units。

因此它不是局部 wall transition，也不满足 RC4.3.0 冻结的显式物理宽度合同。

## L05：molecular-field space 身份

分别构造 `molecular_field_linear_space = spectral` 和 `physical` 的真实 lifted runtime。两个请求都实际调用 physical H assembly/projection 一次，且下列结果完全一致：

- raw H SHA-256；
- projected H SHA-256；
- Q/u/p state SHA-256；
- transform counts（forward `6`、inverse `23`）。

源码 guard 为 lifting 存在时无条件进入 physical branch，runtime metadata 没有单独的 effective molecular-field space。因此 `spectral` 请求当前被静默忽略。

## 对 B8、B20、N4 的裁决

- B8：当前 DST remainder representation 不支持已声明的非零 wall molecular-field convergence；需要 public capability restriction。
- N4：twisted velocity 只有约一阶，并且 molecular-field space 请求被静默忽略；需要 fail-closed contract。
- B20：whole-slab sine taper 不能作为未命名默认策略继续存在；它只能变成显式 opt-in compatibility policy，或由带物理宽度的局部 transition / fail-closed validation 取代。

## RC4.3.2 应冻结的合同

1. 当前 tensor-product DST/DCT backend 上，Plane complete-stress Beris–Edwards 的 static two-sided nonhomogeneous Q Dirichlet 组合必须在 compile/lowering 阶段 fail closed，除非后续提供通过本诊断的表示修复。
2. public error 必须明确说明 unsupported wall molecular-field semantics 和当前受支持子集；不得 warning-only、近似执行或 runtime fallback。
3. `molecular_field_linear_space` 的 requested/effective 必须一致；不能再把 `spectral` 静默执行成 physical。
4. whole-slab sine taper 若保留，只能作为带名称、版本、公式和适用范围的显式 opt-in compatibility policy；不得继续作为隐式 fresh-state default。
5. 本裁决不改 checkpoint identity、生产默认值、PSSolver-Control 或 nematics3d，也未提交 H100。

RC4.3.1 到此完成；它授权进入 RC4.3.2 contract planning，但不授权实现、push、merge、default promotion 或 H100 submission。
