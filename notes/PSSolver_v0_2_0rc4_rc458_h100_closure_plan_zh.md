# PSSolver v0.2.0rc4 RC4.5.8：单次 H100 累计关闭计划

RC4.5.8 只关闭 B11 的 CUDA/accelerator-global-state 剩余门禁，同时对 RC4.5 的八条 runtime family 做累计非回退验证。它不修改数值方程、模型、边界、checkpoint schema、生产默认值或 PSSolver-Control 算法。

冻结身份：

- A baseline：`51f47e9f06bd3e019d785305bff47c80f5519b10`（RC4.5 之前的 RC4.4.7 关闭提交）；
- B runtime candidate：`a195bd05992bc6553c72c400587d2b07afe722d0`（RC4.5.6 最后一个 runtime 实现提交）；
- Consumer qualified source：`b8539e0ec4c137a6c13746f6832e26e74be24515`；
- qualification support 只能位于包含 RC4.5.7 recovery 的后继提交。

单个 H100 Job 必须按顺序执行：

1. 验证 installed wheels、H100、唯一 `cuda:0`、TF32 原始状态与显式 policy、独立 compiler cache、无 source shadow；
2. 在 H100 上真实运行七个冻结 CUDA-only node ID；
3. 对三组 caller global state 验证 construction 无副作用、成功/异常精确恢复、nested LIFO、非法 policy 无副作用；
4. 对 Plane legacy/compiled、Periodic complete-stress、Channel legacy/compiled/complete-stress、Periodic functional、Channel functional 八条 family 执行作用域身份门禁；
5. 执行 96 个 A/B profile：8 family × 2 grid × 3 trial × 2 provider；
6. 运行 Consumer installed-wheel H100 完整 suite；R5 benchmark 仍不在授权范围；
7. 运行 fail-closed analyzer，全部通过后才可写 `COMPLETE`。

性能门限冻结为：每个 family/grid 的 mean 与 median B/A 不超过 1.03，任一 paired trial 不超过 1.05，peak allocated/reserved 不超过 1.03；配对初态和终态 SHA 必须相同，transform count 必须精确，不能出现 fallback、graph break、NaN 或 Inf。

Qualification-support 实现前的构造审计修正了初版机器可读矩阵中的 dtype 误写：`channel_legacy` 与 `channel_compiled_v2` 必须沿用已资格化的 P7.3 float32 构造合同；其余六条 family 使用 float64。该修正不改变 family、grid、trial、门限、方程、默认值或数值实现，防止把外部 harness 的统一 dtype 假设误报为 runtime failure。

同一轮 tiny CPU 单步拓扑审计分别在 A `51f47e9` 与 B `a195bd0` 上重放真实 runtime，确认两边计数完全一致：Plane 为 7/32，legacy/compiled Channel 为 11/36，Periodic complete-stress 与 Periodic functional 为 7/46，Channel complete-stress 为 15/42，Channel functional 为 15/57（forward/inverse）。机器可读计划据此冻结各 family 的真实计数，不再把 Plane/legacy Channel 的旧计数套用到其他路径。

本计划只冻结合同。计划提交当时 Provider 与 Consumer 分支均未满足远端身份前置条件；后续 Consumer qualified source `b8539e0` 及 evidence-only tip `0e3895d` 已推送并验证，但 Provider qualification support 尚未 push，用户也尚未授权 H100 submission，因此不得生成或提交正式 Slurm Job，也不得自动重试。

本地计划门禁已通过：7 个 CUDA-only node ID 均可准确收集，6 个定向合同测试通过；完整 PSSolver 套件为 3368 passed、1 个冻结 CUDA-only deselect、8 个 subtests passed，0 failed、0 skipped。
