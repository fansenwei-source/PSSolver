# PSSolver v0.3 S-09：S-06 Channel CPU 性能预检

分类：`STOPPED_S09_AT_S06_CPU_PERFORMANCE_PREFLIGHT`。

本地预检使用 v0.2.2（`2a1d0f7`）作为 baseline，S-09 runtime
`f94b9be` 作为 candidate。在冻结的 Channel `128×20×20`、长度
`40×6×6` 上，对 legacy、compiled-v2、complete-stress 和 functional
的 tolerance/fixed-work 两种模式分别运行 5 个交替配对 trial。每个
隔离子进程 warmup 3 步，再计时 30 步，并在同一窗口采样 RSS。

全部内存门禁通过。timing 有四项失败：

- legacy/tolerance：中位比 `1.033022`；
- legacy/fixed-work：最大配对比 `1.086032`；
- compiled-v2/tolerance：中位比 `1.030973`，最大 `1.050017`；
- complete-stress/tolerance：中位比 `1.050255`，最大 `1.137853`。

functional 两种模式、compiled-v2 fixed-work 和 complete-stress
fixed-work 通过。所有样本均保留，没有删除或替换离群值。门限没有
放宽。

依据用户预先冻结的 fail-fast 要求，S-09 停止在 CPU 门禁；没有继续
Provider 全量资格、R-0022、H100 helper 或 H100 submission。该结果不
证明 CUDA/H100 一定失败，但已经构成提交 H100 前必须先处理或由用户
重新决策的性能风险。

原始矩阵、80 个 cell JSON、环境记录和绑定测试日志保存在：

`/home/fansenwei/pssolver_v0_3_s09_cpu_preflight_f94b9be_20261023_v1`

该目录中的 `COMPLETE` 只表示失败证据收集完整，不表示 S-09 资格通过。
