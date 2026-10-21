# PSSolver v0.2.1 N-1 真实 NFS CPU 资格记录

日期：2026-10-20

最终分类：`PASS_V0_2_1_N1_REAL_NFS_CPU_QUALIFICATION`

本记录是 `maint/v0.2` 上 N-1 候选实现的后继证据记录。已验证的源码提交为
`15670eb264a44c6707aeb1695b74e6cdbb34f611`，其中运行时实现提交为
`927dc89aa63bcee9b03248f52662389bc947c47a`。原本记录本地候选状态的
`notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json` 保持不变。

## 已完成门禁

- 从 clean detached worktree 构建 source archive 和 wheel。
- 在固定 HPCC Python 3.10.18 环境中新建 venv，以 `--no-deps` 安装 wheel；`pip check` 通过，无源码 shadow import。
- installed-wheel N-1 fault-injection tests：13 passed，0 failed，0 skipped。
- `/home/fansenwei` 与 `/scratch1/vincent` 均实测为 NFS，`statfs` 类型为 `0x6969`。
- 两处 NFS 上的 `RENAME_NOREPLACE` 原始路径均以 `EINVAL`（errno 22）拒绝，随后只在已确认的 NFS 上进入受限 fallback。
- 两处 NFS 的目录 fsync、首次发布、重复目标拒绝、无临时残留均通过。
- node-local fast path 与 NFS fallback 生成的 checkpoint tree 逐字节一致。
- 真实 Periodic functional checkpoint 在两处 NFS 上导出、重新导入并精确恢复；provider identity 保持不变。
- 双节点 CPU Slurm Job `10902631` 在 `compute-9-0` 与 `compute-9-1` 上并发发布同一目标，精确得到 1 个成功和 1 个 `CheckpointExistsError`，最终 tree 匹配胜者且没有 staging 残留。

## 归档

控制目录：

`/home/fansenwei/pssolver_v021_n1_nfs_cpu_15670eb_20261020_v1`

scratch 目录：

`/scratch1/vincent/PSSolver/data_pssolver_v021_n1_nfs_cpu_15670eb_20261020_v1`

控制目录 checksum：29/29 PASS；manifest SHA-256 为
`9748645f3690e9681d2ba5ea7cf0f85d81aadce4ca92a80aca3d3cd19213822b`。

scratch checksum：7/7 PASS；manifest SHA-256 为
`5c06f84993367c856f9f72d7979494315fc0efbc1f3e4160606dcfdce303d031`。

## 边界与下一步

这不是 H100 资格验证。`formal_h100_submission_count = 0`，没有申请 GPU，也没有修改
PSSolver-Control、nematics3d、checkpoint schema、Functional API 版本、provider identity、数值算法或生产默认值。

当前 CPU 阶段已经完成，但 `n1_complete = false`。下一步需要单独授权一个小型 installed-wheel H100 checkpoint/restart 资格任务；通过后才能进行最终 v0.2.1 版本号、打包、tag 与 Release 收口。
