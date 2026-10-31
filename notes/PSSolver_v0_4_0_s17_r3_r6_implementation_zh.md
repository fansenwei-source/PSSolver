# PSSolver v0.4 S-17：R3 / R6 本地实现候选

本候选依据已确认的 D-0017，只在本地 `dev/0.4` 实施，等待 R-0050
独立复核；没有 push、GPU 或 H100 操作。

## R3

`pssolver.functional.api` 继续提供原样的 Functional API 1.1，46 个稳定
导出及语义均不变。`pssolver.functional` 的 85 个兼容导出在整个 v0.4
周期继续可用，但首次访问旧名称时发出 `DeprecationWarning`；单独导入
`pssolver.functional.api` 不发告警。旧 facade 的 raw builder 仍返回原实现，
不会被稳定 facade 的包装实现替换。可以安全共享的合同、异常和版本对象
则从稳定 API 解析，减少重复所有权。

Control-first 边界未变：Functional API 没有变化，`beris_edwards.py` 与
`q_tensor.py` 的 git blob 仍分别为 `07f1f03f...` 与 `5427df44...`，因此
本候选不需要再次迁移 Control main。以后若改变这些边界，仍须先完成隔离
的 Control main CPU 迁移；`research/3d-warmup` 不在本轮范围。

## R6

新增明确的 `pssolver.experimental.boundary_pilots` 入口，统一列出 Robin、
finite-Q 和 lifting 的 59 个声明、规划、算子、运行时与工作流对象。原来的
`planning`、`operators`、`runtime`、`workflows` package facade 名称在 v0.4
中继续可用，但首次访问时提示改用 experimental 入口。原定义模块路径和
已发布 checkpoint reader 不移动、不改 schema。

公开 `Simulation` 的 Robin 执行仍然 fail-closed；lifting 在 B8/S2-07
解决并通过独立资格前仍关闭。本候选没有改数值算法、GPU 专用路径、门限、
生产默认值、skip 或 xfail。
