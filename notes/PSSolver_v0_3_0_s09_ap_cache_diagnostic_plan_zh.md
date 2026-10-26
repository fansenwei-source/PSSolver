# PSSolver v0.3.0 S-09 A·p 缓存诊断计划

状态：`READY_FOR_R0028_REVIEW`。本计划依据 D-0012，仅用于解释 V-41 的
H100 reserved-memory 增量；它不是资格测试，不能将 S-09 判为通过。

## 固定范围

- runtime：`85ba3aef75548d549d9db453ca4a5ab604b17a9d`；
- baseline：v0.2.2，`2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70`；
- family：`channel_legacy`、`channel_compiled_v2`；
- grid：`channel_R128x20x20`、`channel_R512x40x40`；
- role：baseline、candidate cache-on、candidate cache-off；
- 每个 family/grid/role 6 次 trial，每个 profile 60 个正式样本，共 72 个
  独立子进程；三种 role 按 trial 循环换序；
- 每个子进程使用独立且初始为空的 Triton/TorchInductor cache；
- 保留全部样本，并报告冻结分析器使用的总体均值比、profile 均值中位数比、
  六次 trial 比值中位数、两种相对顺序的中位数、顺序层差异以及 peak
  allocated/reserved 比值。

原 1.03/1.05 门限只作为参照原样展示；诊断分析器不据此给出 PASS/FAIL，
也不改变冻结的资格计划、案例、样本或门限。

## cache-off 实现边界

cache-off 只存在于诊断 profile 子进程：临时覆盖
`ChannelNoSlipModalStokesSolver._store_pressure_guess` 与
`_cached_pressure_operator_action`，保留 `pressure_guess`，但不保存也不复用
A·p。进程结束前恢复原方法。每个 cache-off profile 必须记录：缓存读取次数、
被抑制的保存次数、每次保存后四个缓存字段均为 `None`。未获得这些证明时，
profile 失败。

Provider wheel、runtime、冻结计划和阈值均不修改。

## 本地门禁

在 D-0008 指定的基线 venv 硬链接克隆中，以 installed wheels 运行完整的
72-process CPU rehearsal。rehearsal 使用两个小网格、每 profile 1 步 warm-up
与 2 步正式执行，仅验证编排、隔离、身份、cache-off 证明和状态一致性，不产生
性能结论。candidate cache-on/off 的初态与末态摘要必须逐字节相同。

本地证据必须包含环境、`sys.path`、wheel 哈希或 RECORD 摘要、pytest 日志、
JUnit、rehearsal 日志、结果、`checksums.sha256`、`COMPLETE` 和清理 venv 后的
`PRUNED.txt`。

## 结果解释

结果只回答：关闭缓存后，计时和 peak allocated/reserved 如何变化。它不判定
S-09 通过。R-0028 独立复核 PASS 后，才起草一次诊断 H100 授权；实际诊断数据
仍需用户决定是撤回缓存后重跑完整 S-09，还是接受显存例外。
