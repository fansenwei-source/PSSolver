# S-09 A·p 缓存诊断 R-0028 候选

分类：`READY_R0028_S09_AP_CACHE_DIAGNOSTIC_REVIEW`。

本候选依据 D-0012，只准备一次非资格 H100 诊断。运行时仍为
`85ba3aef75548d549d9db453ca4a5ab604b17a9d`，两个 Provider wheel 不变；
qualification-support 固定在
`b8df25e56ed092cc732ef1dbfba591c2af776d1a`，其 archive 不含
`pssolver/` 或 `pssolver_control/`。

计划包含 72 个独立 profile：两种 Channel runtime、两个网格、三种 role、六次
trial。每个 H100 profile 保留 60 个正式样本，使用独立空编译 cache，三种 role
循环换序。分析器报告冻结分析器的统计量，但不使用门限给出资格判定。

cache-off 只在子进程 harness 临时禁用 A·p 的保存和读取；压力 warm start 保留。
每个结果都必须证明至少一次缓存读取被强制 miss、至少一次 A·p 保存被抑制，且
保存后缓存字段全部为 `None`。

installed-wheel CPU 演练从 D-0008 的 v0.2.2 基线 venv 以 `cp -al` 建立两个
临时环境。baseline 从自身 site-packages 导入 v0.2.2，candidate 从自身
site-packages 导入冻结的 v0.3 wheel；support 路径不含包源码。72/72 profiles
完成、全部有限、全部 cache-off 证明通过，candidate cache-on/off 的初态和末态
摘要逐字节相同。CPU 计时不作性能解释。

第一次编排尝试在任何 profile 前发现 runner 错误拒绝正常的 venv Python
符号链接；失败日志保留。支持提交 `b8df25e` 只修正该外部门禁，完整矩阵随后只
重跑一次并通过。

本记录没有授权 push、HPCC、Slurm 或 H100，也没有把 S-09 判为完成。R-0028
PASS 后，才向用户提供带完整哈希的一键 H100 诊断授权草稿。
