# PSSolver v0.2.0rc5 G-1.8：HPCC 九阶段计划冻结

G-1 的真实九阶段 H100 计划已经在 HPCC 登录节点冻结并重复通过 preflight，但本轮没有 push，也没有提交 H100 Job。

冻结计划依次执行 `environment`、`cuda_only`、`tf32_scope`、`checkpoints`、`numerical`、`profiles`、`consumer`、`assemble` 和 `analyze`。每个阶段只绑定一个绝对 Python 路径和一个普通、非符号链接 Python 脚本；两者的实际文件 SHA-256 已写入冻结计划。计划不包含 shell 命令字符串或内联 Python，所有正式产物均限制在最初不存在的 H100 输出根目录内。

新的 source-free Provider support archive 绑定提交 `fa8dc94f7b0d4e97999ad36c94491a5b9c84b89c`，解包目录不含 `pssolver/`，因此不会遮蔽三个 installed-wheel 环境。冻结计划 SHA-256 为 `a4293716b6f916aa0a17bacc88df2693638ce34891dea7a00989b1f431111592`。九个 stage producer 均从 `/tmp` 完成直接 `--help` 导入 smoke。

96 项 profiler 子计划也已在登录节点展开：序号为 1–96，键 96/96 唯一，全部使用 profiler 正式接受的 `--device cuda`，没有 `cuda:0` 参数，也没有任何既有 profile 输出。子计划 SHA-256 为 `652ed098ad5d1f73d414e874be2c1e1d29e9ba154d2df8fcd6258749dce32795`。

最终检查后，正式输出目录 `/scratch1/vincent/PSSolver/data_pssolver_rc5_g1_h100_674a812_20261015_v1` 仍不存在；正式 H100 submission 数量仍为 0。此记录只说明计划已具备可执行且可审计的提交前状态，不是 G-1 qualification PASS。

下一步需要两个彼此独立的明确授权：先 push `fix/v0.2.0rc5`，再授权恰好一次 G-1 H100 submission，并继续禁止自动重试。未获得授权前不得调用 `sbatch`。
