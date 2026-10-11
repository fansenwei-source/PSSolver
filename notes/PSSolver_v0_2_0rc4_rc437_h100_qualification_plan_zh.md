# PSSolver RC4.3.7 单次 H100 lifting 关闭计划

RC4.3.7 将 `1b15bf8` 固定为 RC4.2 baseline，将 `e1dea73` 固定为 RC4.3 runtime candidate。后续 qualification-only 提交可以承载 runner、analyzer、测试和本计划，但不得再修改 runtime source。

本阶段不会重新开放 public lifted Plane。RC4.3.3 已确认当前张量积基不能诚实实现非零壁面分子场，因此公共 `Simulation` 入口仍须在构建 runtime 前以稳定错误码拒绝。H100 correctness 只关闭 field-neutral static lifting、localized fresh conditioning、fresh-only provenance 和 same-version restart 这些内部基础设施。

性能与显存比较使用仍受支持的 homogeneous-Neumann `legacy_production` 路径：R128/R320、A/B 各三次、10 步 warmup 和 50 步正式计时。两端必须使用相同冻结初态，并且初态与最终状态 SHA 跨版本完全一致；forward/inverse transform 必须保持 7/32，compiled execution 必须 effective，runtime/compile fallback 与 graph break 必须为零。candidate/baseline 的 mean 和 median timestep ratio 不超过 1.15，peak allocated 与 peak active ratio 不超过 1.10；reserved 只作 allocator 诊断。

candidate-only CUDA 门禁使用 localized quintic、2.0 物理长度过渡宽度，验证 CPU/CUDA relative L2 不超过 `1e-15`、过渡层外逐字节不变、过渡层确实发生变化、wall residual 不超过 `1e-12`、fresh provenance 完整且不进入 restart compatibility identity，以及 4-step continuous 与 2+2 file restart 逐字节一致。

H100 必须另行明确授权。冻结配置为 `hagan-gpu` / `hagan-lab` / `medium` / `gpu:H100:1`，只允许一个正式 Job，不得自动重试。当前工作只实现并验证本地 qualification support，不调用 `sbatch`，不 push，不修改 PSSolver-Control、nematics3d 或生产默认值。

本地资格支持现已通过 53 项定向测试；完整 CPU suite 为 3035 passed、7 个精确 CUDA-only deselect、8 subtests passed，零 failure、skip 和 xfail。当前状态是等待独立 H100 授权，而不是 RC4.3.7 已完成。
