# PSSolver rc5 G-1 measurement-window 生命周期恢复 v10

## 当前分类

`READY_FOR_EXPLICIT_H100_AUTHORIZATION`

v9 唯一 H100 Job `10885954` 在第一个 profile 完成计算、进入结果对象组装时，因为 qualification-only profiler 已执行 `del case` 却继续读取 `case.configuration_identity` 而停止。该事件不支持 solver、科学、数值、CUDA、性能或显存失败。

## 修复

修复只作用于 `benchmarks/profile_rc458_runtime_family.py`：在删除正式 runtime 前复制并持久化 `configuration_identity`，结果对象使用该持久值。正式 runtime 仍会在 fresh-runtime diagnostic replay 前释放，因此没有撤销原显存隔离设计。

新增回归测试通过受控 runtime 实际执行 corrected-window 生命周期。旧代码会在同一路径触发 `UnboundLocalError`；修复后要求配置身份、正式 completed steps 和 diagnostic replay 均正确返回。

## 不变边界

- 不修改 PSSolver production solver/runtime；
- 不修改科学方程、默认值或任何门限；
- 不改变 `10 warmup + 3 untimed rewarm + 60 timed steps`；
- 不改写 v9 实现记录或 v9 失败归档；
- 当前没有 H100 授权，不提交 Job、不自动重试。

## CPU / synthetic 门禁

- 本地 source compile：PASS；
- HPCC Python：3.10.18；
- 固定提交：`305890013831a9eb79ac1ef671c88e3e5672398c`；
- 完整历史 bundle：`075491e4aed66761ac54103fc87de33af543b3afc8dbbf8f9b8c311d36b2efac`；
- 定向 CPU/synthetic replay：71 passed，0 failed，0 skipped，0 xfailed（8.04 秒）；
- replay 后 worktree status、ignored status 和缓存计数均为 0；
- H100 未执行。

## 下一门禁

该 qualification-only 修复已经达到等待新的明确 H100 submission 授权的状态。本记录不授予该授权，也不把 G-1 判为完成。
