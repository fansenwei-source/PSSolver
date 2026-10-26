# S-09 A·p 精确撤回代理 R-0029 候选

分类：`READY_R0029_S09_AP_EXACT_REVERT_PROXY_REVIEW`。

本候选按 R-0028 返工，只修改诊断支持代码。运行时仍为 `85ba3ae`，baseline
与 candidate wheel 哈希不变，没有加入第四种折中角色。

核心变化：

- 在候选模块命名空间中精确编译 `f94b9be` 的 691 行 `_solve_pressure`；
- 72 项矩阵改为 baseline、cache-on、revert-proxy 三角色；
- 六次 trial 使用全部六种排列，每对角色前后顺序 3:3；
- 单 trial 比值与冻结分析器一样使用 `_profile_center`，不再使用样本均值；
- R512 cache-on 与 revert-proxy 另各做一次不计时 CUDA allocator snapshot。

安装版 CPU 演练 72/72 完成，24 个代理证明和 24 对状态逐位一致性全部通过。
R512 CPU 活跃内存峰值为：cache-on `332723328`，revert-proxy 与直接
`f94b9be` 源码均为 `326169728` bytes；三者终态逐位相同，代理与直接源码的
A·p 调用次数也一致。差值 `6553600` bytes 正好是一个压力场。

本地归档：

`/home/fansenwei/pssolver_v0_3_s09_ap_revert_proxy_r0029_local_b031421_20261026_v2`

共 331 个 checksum 条目，manifest SHA-256 为
`fe9c9ba3742c297ea7091d4733d1a0d94a74efc496450288a3d0b65e068eab69`，
`COMPLETE` 存在；定向测试 29 passed，JUnit 已列入清单，临时 venv 已清理。

这是 analysis-only 候选，不授予 push、HPCC、Slurm 或 H100 权限，不把 S-09
判为完成，也不接受显存例外。下一步仅是把 R-0029 交给 Claude 独立复核。
