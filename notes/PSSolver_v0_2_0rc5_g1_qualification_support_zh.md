# PSSolver v0.2.0rc5 G-1.5：版本化单 H100 资格工具

本阶段已经完成 rc5 G-1 的版本化 qualification-only 工具，但没有 push，也没有提交 H100。Provider 数值实现、PSSolver-Control、nematics3d、生产默认值、checkpoint schema 和公开 API 均未修改。

新增工具冻结了 96 项 A/B profile matrix、48 项 TF32 状态恢复矩阵、7 个 CUDA-only tests、checkpoint、八条 runtime-family 数值门禁、完整 Consumer 门禁及最终累计 analyzer。所有子进程都使用 argv 列表而非 shell 字符串；A/B venv 入口保持 lexical identity；每个 profile 使用独立 Triton/TorchInductor cache；阶段进度和最终 JSON 都原子写入。

最重要的修正是移除了 RC4.5.8 analyzer 的全局 final-state SHA 假设。Plane 与 Channel 的六条未变路径仍要求 A/B 字节一致。`periodic_complete_stress` 与 `periodic_functional` 因 rc5 主动修复 stored self-conjugate planes，不得再用 rc4 谱状态逐字节相等作为 oracle；它们必须分别对 Q、velocity、pressure 满足 relative-L2 和 Linf 不超过 `1e-12`，并且 rc5 candidate 的 Hermitian violation 必须精确等于零。

CPU 合成验证覆盖了完整 96-profile 正例，以及非 Periodic SHA 不一致、Periodic 物理误差、非零 Hermitian 破缺、TF32 未恢复、Consumer identity 不一致、性能回退和 reserved-memory 回退等负例。20 项测试全部通过。

当前仍不能提交 H100。下一步是基于 HPCC 上的 fresh installed-wheel venv 和确定的只读证据路径生成一份九阶段 command plan，执行 `--preflight-only`，核对 Provider/Consumer 远端 tip 与 Slurm 路径，然后再单独请求 push 和唯一一次 H100 submission 授权。
