# PSSolver rc5 G-1 测量窗口修正 v9 预注册

## 结论

本步骤只冻结 qualification harness 的测量窗口修正合同。它没有修改求解器、生产 runtime、科学方程、默认值或任何既有性能阈值，也没有提交 H100 Job。

当前分类为 `READY_FOR_QUALIFICATION_SUPPORT_IMPLEMENTATION`。G-1 仍未完成，v7 的失败分类保持不变，v8 仍只是 analysis-only 诊断。

## 修正后的正式窗口

每份 profile 使用完全相同的阶段顺序：

1. 构建真实 runtime；
2. 运行 10 个既有 warm-up steps；
3. CUDA synchronize；
4. `gc.collect()`；
5. `torch.cuda.empty_cache()`；
6. 记录 flush 后的显存快照；
7. 运行 3 个不计时的 post-flush rewarm steps；
8. CUDA synchronize，并保存 cold-recovery 诊断；
9. 重置正式显存峰值、transform timer，并取得正式 compiler-counter 起点；
10. 使用 CUDA events 计时完整的 60 steps；
11. 保存全部 60 个原始样本和正式结果。

因此正式状态窗口从原来的第 11–70 步变为第 14–73 步。baseline 与 candidate 必须使用相同阶段顺序。

3 个 rewarm steps 不属于正式 timing、transform counter、compiler counter 或 steady-state memory peak 窗口。它们产生的 cold allocation recovery 显存信息不能丢弃，只能作为独立 diagnostic 保存。

## 不变的裁决合同

- 48 份 profile，4 个 focused cells，每 cell 6 个配对 trials；
- 每份 profile 的 60 个正式样本全部参与 mean、median、sample standard deviation 和 CV；
- `CV <= 0.05`；
- 不允许删除第一样本、删除 outlier 或替换 trial；
- mean ratio、median ratio、每对 ratio、allocated/reserved memory ratio 的上限仍分别为 `1.03`、`1.03`、`1.05`、`1.03`、`1.03`；
- graph break 必须为 0，runtime/compile fallback 必须为 false，transform raw counts 必须精确。

## 剩余波动诊断

v8 已证明第一样本污染不是全部解释。因此 v9 还冻结了一个独立、全新 runtime 的 diagnostic replay。它必须与正式 profile 使用相同 seed、配置和初态 SHA；按适用性记录公开 pressure iteration/residual/termination diagnostics、窗口外设备遥测和分阶段显存快照。

该 replay 不能用于性能或显存裁决。若某项公开诊断不适用，必须写明 `applicable=false` 和原因，禁止填造数值占位符。

## 后续边界

下一步只实现 qualification-only profiler、runner、analyzer 和 synthetic tests，并完成 CPU preflight。实现通过后，仍需用户另行明确授权，才能提交新的 H100 recovery Job。
