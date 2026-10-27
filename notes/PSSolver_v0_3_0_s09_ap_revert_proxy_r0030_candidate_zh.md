# S-09 R-0030 A·p 诊断返工候选

分类：`READY_R0030_S09_AP_DIAGNOSTIC_REWORK_REVIEW`。

本候选只修改 qualification support 和记录；运行时仍是 `85ba3ae`，v0.2.2 与
v0.3 wheel、72 项矩阵、案例、样本、冻结统计和阈值均不变。

R-0029 的两项返工已完成：

1. 诊断作业复用主 S-09 的 GCC/G++ 7.3.0、CUDA 12.1、节点本地 `TMPDIR` 和
   空 cache CUDA `torch.compile` 预检，且在矩阵前持久化编译环境；
2. 撤回 CPU 活跃峰值 parity 证明。重复运行会波动一至两个压力场，因此该探针
   只作参考。代理可信度改由 `f94b9be` 方法 SHA、候选命名空间编译、空缓存字段、
   78 次算子调用与终态逐位一致来证明。

V-41 的增量仍为一个压力场，且由 `2573dec` 引入；究竟是持久缓存还是局部 A·p
生命周期落在 CUDA 峰值路径上，留给两次不计时的 H100 allocator snapshot。
V-41 保持 `registered`，S-09 未完成。

本地 qualification-support 门禁为 48 passed、0 failed、0 skipped，日志与 JUnit
已归档。此记录不授权 push、HPCC、Slurm 或 H100；R-0030 PASS 后才可按精确绑定
值起草一次性诊断 H100 授权。
