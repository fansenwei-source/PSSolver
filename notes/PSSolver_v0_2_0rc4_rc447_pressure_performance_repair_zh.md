# PSSolver RC4.4.7 Channel pressure 性能修复记录

## 状态

本记录只登记 v8 H100 证据暴露出的 Provider 运行时性能问题及本地修复。当前没有授权 push，也没有授权新的 H100 作业，因此不能把本记录解释为 RC4.4.7 资格通过。

## v8 的有效结论

Job 10878190 完成了 7 个 CUDA 测试、9 个正确性门禁和 12 个 profile。候选与基线的压力迭代历史和最终状态一致，未出现 CUDA、有限性、科学或数值失败；但是候选在 R128 和 R512 上分别慢约 2.80 倍和 2.62 倍。

这不是测量噪声。逐项检查生产 pressure PCG 后确认，scale-safe 实现把若干本来可合并的标量统计拆成了连续 GPU→CPU 同步，并在每次迭代中重复计算不变的 RHS RMS。数学公式正确，但同步调度使小型标量控制路径主导了 timestep。

## 修复边界

本次修复：

- 缓存活跃 pressure 自由度数量；
- 从已经测得的 scale-safe RHS magnitude 派生 working-scale magnitude；
- 复用 PCG 更新已经需要的内积；
- 合并 residual magnitude、breakdown 输入和 finite 标记的普通路径观察；
- 保留公开 scale-safe primitive 作为极端尺度的 fail-closed 回退。

没有修改 pressure policy、固定工作量、容差、breakdown 定义、诊断 schema、模型方程或生产默认值。

## 本地验证

在 CPU 上，对 fixed-work 与 tolerance 两种模式、batch 1 和 batch 2，修复前后的 pressure tensor、迭代次数和完整 `last_pressure_diagnostics` 均完全相同。

RTX 3060 Ti 上的隔离筛查中，单次 pressure solve 从失败候选的约 20.25 ms 降至约 6.80 ms，基线约 6.89 ms；交错 R128 全 profile 的候选/基线也回到约 1.0。GPU 型号与正式环境不同，所以这些数据只能证明修复方向值得进入 H100 复核，不能代替 H100 资格证据。

## 后续边界

下一步应先完成 CPU 门禁和独立 Provider commit；随后以该 commit 为新候选，更新 qualification-only profile/analyzer 支持并再次审查。只有用户另行明确授权后，才允许 push 或提交一次新的 H100 recovery，且不得自动重试。
