# PSSolver RC4.3.0 lifting correctness 规划

RC4.2 已由 commit `1b15bf8` 和 H100 closure 正式关闭。RC4.3 只处理 B8、B20、N4：lifted Plane 的壁面分子场表示、被忽略的 `molecular_field_linear_space`，以及 fresh initial state 的全域正弦 taper。本文件完成规划，不修改求解器，也不授权 H100。

## 为什么不能直接“再换一次基”

现有证据表明问题不是 metadata。非零壁面分子场的误差在 `nz=16,32,64,128,256` 上约为 `0.037,0.034,0.032,0.030,0.030`，没有趋零；真实 lifted solver 中的 elastic Laplacian 仍受齐次 DST 表示约束。扭转剖面的速度收敛率也从约 `1.0` 降为 `0.75`。同时，配置写 `spectral` 或 `physical` 得到逐字节相同结果，说明请求值没有真正控制执行路径。

因此 RC4.3.1 必须先做 analysis-only 裁决，并使用真实 solver 和独立物理空间参考。只调用投影器、只改 metadata、把 `spectral` 静默改成 `physical`、或者用 warning 继续运行，都不能关闭问题。

## RC4.3.1：先判断“可修”还是“必须限制”

固定检查五类用例：既有 affine manufactured control、非零壁面分子场、扭转 Q 到速度、fresh uniform/defect-gas 初态，以及 `spectral`/`physical` 请求与 effective path 身份。壁法向分辨率固定为 16、32、64、128、256。

参考不能复用待测 lifting projector。优先使用解析导数；否则使用物理 Q 上的二阶有限差分和一份独立 direct constitutive evaluation。必须保存壁面与第一内部单元误差、bulk H、速度、去均值压力、相邻/回归收敛率、transform 数量和 fallback。宣称二阶的量要求回归阶至少 1.8，最后两个 wall-sensitive error ratio 必须严格下降。

最终只能得到两种分类：

1. `REPRESENTATION_REPAIR_SUPPORTED`：真实 solver 的壁面 H 收敛，扭转速度不退化，请求/effective space 一致；
2. `PUBLIC_CAPABILITY_RESTRICTION_REQUIRED`：若当前张量积基无法诚实实现，则 compile/lowering 必须在 runtime 构建前拒绝精确定义的组合，错误消息说明不支持的壁面语义和仍支持的子集。不得 approximate、fallback 或只移除测试。

## B20：fresh 初态调理

优先方案是有明确物理宽度和光滑公式的局部壁面过渡层；还要比较“无自动 taper、改为 fail-closed 初态兼容检查”和“全域正弦 taper 仅作为显式 opt-in”两种方案。选定方案必须满足：

- 公式、宽度、单位、作用分量和策略版本进入 canonical metadata；
- 若采用局部层，层外请求的 Q 必须逐字节不变；
- 宽度具有网格无关的物理语义，而不是未记录的 cell 数；
- 只影响 fresh construction，不进入 restart compatibility identity，也不改写恢复态；
- 未启用 lifting 的初态与 RC4.2 逐字节相同；
- CPU/CUDA 确定性一致。

## 分阶段顺序

1. RC4.3.1：analysis-only 表示与初态裁决；
2. RC4.3.2：冻结 repair/restrict、effective-space、初态策略和版本合同；
3. RC4.3.3：修复分子场表示，或实现 fail-closed 公共能力限制；
4. RC4.3.4：修复/显式化 fresh conditioning；
5. RC4.3.5：runtime 与 checkpoint 集成；
6. RC4.3.6：source + installed-wheel CPU closure；
7. RC4.3.7：单次 H100 correctness/restart/performance/memory closure。

RC4.3.7 延用已经证明可执行的 lifting 门限：R128/R320 各三次配对，mean/median timestep ratio 不超过 1.15，peak allocated/active 不超过 1.10，reserved 只作 allocator 诊断，transform 数保持 control 7/32，fallback 和 graph break 为零。若动力学改变，跨版本容差必须在运行前另行冻结；同版本 continuous/restart 仍必须逐字节一致。

## 架构边界

field-neutral lifting core 保留，Q 特化只留在 Plane Beris–Edwards composition/runtime。RC4.3 不增加非齐次 Neumann、Robin、新 geometry 或新 model；不修改 PSSolver-Control、`nematics3d`、生产默认值或历史证据。

当前只完成 RC4.3.0。RC4.3.1 analysis、任何实现、H100、merge、默认值提升和发布都需要后续单独授权。
