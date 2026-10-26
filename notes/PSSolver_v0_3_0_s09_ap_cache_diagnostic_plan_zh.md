# PSSolver v0.3.0 S-09 A·p 撤回代理诊断计划

状态：`READY_FOR_R0029_REVIEW`。本计划依据 D-0012，并按 R-0028 的返工意见
修订；它只解释 V-41 的 H100 显存增量，不是资格测试，不能将 S-09 判为通过。

## 固定范围

- runtime：`85ba3aef75548d549d9db453ca4a5ab604b17a9d`；
- baseline：v0.2.2，`2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70`；
- family：`channel_legacy`、`channel_compiled_v2`；
- grid：`channel_R128x20x20`、`channel_R512x40x40`；
- role：baseline、candidate cache-on、candidate revert-proxy；
- 每个 family/grid/role 6 次 trial，每个 profile 60 个正式样本，共 72 个
  独立子进程；六次 trial 使用三种 role 的全部 6 种排列；
- 每一对 role 的前后顺序为 3:3，每个 role 在第一、第二、第三位置各出现两次；
- 每个子进程使用独立且初始为空的 Triton/TorchInductor cache；
- 保留全部样本。单 trial 比值使用冻结分析器 `_profile_center`：把 60 个样本
  分为六个固定 10 样本块，取六个块均值的中位数，再计算角色间比值；其余总体
  均值比、profile 均值中位数比、顺序层统计与 peak allocated/reserved 统计均与
  冻结分析器一致。

原 1.03/1.05 门限只作为参照原样展示；诊断分析器不据此给出 PASS/FAIL，也不
改变冻结资格计划、案例、样本或门限。

## revert-proxy 实现边界

R-0028 证明仅禁用持久缓存不能代表撤回：在 R512 CPU 活跃张量时间线中，cache-on
和原 cache-off 都是 316,339,328 bytes，而真正的 `f94b9be` 是 309,785,728
bytes。增量来自 2573dec 延长了单次 `_solve_pressure` 内局部 A·p 的生命周期，
而不是持久缓存字段本身。

新 harness 从候选 wheel 的 `_solve_pressure` 源码出发，使用 fail-closed 的精确
逆补丁重建 `f94b9be22fd767fee432caf30f0e5c47aad8e9c3` 的 691 行历史方法。重建前
必须匹配候选方法 SHA-256 `43316e0c…`，重建后必须匹配历史方法 SHA-256
`2ffb5673…`；历史完整源文件 SHA-256 为 `c539275f…`。方法只在候选 wheel 的
`channel_no_slip` 模块命名空间中临时编译，在独立子进程结束前恢复。每个代理
profile 必须证明方法实际执行，且四个 A·p cache 字段始终为空。

Provider wheel、runtime、冻结计划和阈值均不修改。R-0028 的 persistent-cache-off
实现保留为附加诊断工具，但不进入 72 项正式计时矩阵，也不加入未经授权的第四
角色。

## 本地门禁

在 D-0008 指定的基线 venv 硬链接克隆中，以 installed wheels 运行完整的
72-process CPU rehearsal。rehearsal 使用两个小网格、每 profile 1 步 warm-up
与 2 步正式执行，仅验证编排、隔离、身份、撤回代理证明和状态一致性，不产生
性能结论。candidate cache-on/revert-proxy 的初态与末态摘要必须逐字节相同。

另用 R512 `channel_legacy` 做 CPU `torch.profiler` 活跃张量时间线：候选代理的
终态必须与 cache-on 和直接检出的 `f94b9be` 源码逐字节相同，且代理峰值必须与
直接源码峰值一致。CUDA snapshot helper 也在 CPU 上走完同一流程，但明确跳过
CUDA snapshot 文件。

本地证据必须包含环境、`sys.path`、wheel 哈希或 RECORD 摘要、pytest 日志、
JUnit、rehearsal 日志、内存时间线摘要、结果、`checksums.sha256`、`COMPLETE`
和清理 venv 后的 `PRUNED.txt`。

## H100 附加快照

72 项计时矩阵之外，对 R512 `channel_legacy` 的 candidate cache-on 和
candidate revert-proxy 各运行一次不计时的独立进程。进程调用
`torch.cuda.memory._record_memory_history` 与 `_dump_snapshot`，保存 allocator
snapshot，用于定位 CUDA 峰值增量的分配位置。两次运行不进入计时统计，也不
改变冻结门限或诊断结论。

## 结果解释

结果只比较候选与真实撤回代理的计时、peak allocated/reserved 和 allocator
分配时间线。它不判定 S-09 通过。R-0029 独立复核 PASS 后，才起草一次诊断
H100 授权；实际数据仍只用于决定撤回缓存后重跑完整 S-09，或由用户另行决定
是否接受例外。D-0012 本身不接受例外。
