# PSSolver v0.3 S-09 CPU 性能诊断与停止记录

分类：`STOPPED_S09_AFTER_OPTIMIZATION_AT_BOUND_FIXED_WORK_TIMING_SAMPLE`。

两次完整的 v0.2.2 自对照均没有出现单次比值大于 `1.05`，因此没有触发
预先允许的 `6×60` 稳定测法。`05f55d1 → 98742b6` 的 S-06 前后矩阵
8/8 通过；tolerance 中位成本约增加 1–2%。只撤掉 C1-05 接受前真实
残差重算的因果对照，确认这次重算是生产 tolerance 路径稳定开销的来源。

`2573dec` 将接受时已经算出的 pressure operator action 复用于下一次
production warm start。外部替换或原地修改 warm-start tensor 时缓存失效；
跨 2 的幂工作尺度和连续 20 次求解与强制直接重算逐位相同。Functional
zero-start map 没有增加 hidden cache，fixed-work 也不读取该缓存。

优化后的探索矩阵 8/8 通过，但绑定 `2573dec` 的正式矩阵仍有一个样本
失败：legacy fixed-work trial 2 比值为 `1.082258`，该单元格中位数为
`0.996050`。该失败不在优化路径上，但冻结合同要求保留全部样本，所以
S-09 再次 fail-closed。没有放宽门限、删除样本、创建 R-0022、推送或
运行 HPCC/H100。

证据目录：

`/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2`

其中 `COMPLETE` 只表示停止证据收集完整，不表示 S-09 资格通过。
