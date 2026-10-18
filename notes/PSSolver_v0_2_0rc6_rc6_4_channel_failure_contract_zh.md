# PSSolver v0.2.0rc6 RC6-4 Channel 失败异常与运行状态合同收尾

## 结论

RC6-4 / A7-03 已完成本地 CPU 关闭，分类为：

`PASS_RC6_4_CHANNEL_FAILURE_AND_METADATA_CPU_CLOSURE`

本阶段没有 push，没有提交 H100 作业，没有修改 PSSolver-Control、nematics3d、生产默认值、科学方程、压力算法、积分器、变换算法或 checkpoint schema。

## 修复内容

1. `ChannelPressureConvergenceError` 保留原有公共类型与 `details` 映射兼容性，同时正式归入 `SimulationExecutionError` 公共执行异常层级。
2. `FunctionalConvergenceError` 同时保持 functional 异常层级和 Channel pressure / simulation execution 异常语义。
3. 已启动的 Plane 与 legacy Channel workflow 若因公共执行异常停止，会原子地把 `metadata.json` 从 `running` 终结为 `failed`，持久化完成步数和结构化 failure metadata，然后重新抛出原异常对象。
4. 失败路径不会写入 `COMPLETE`，也不会发布非有限数组。
5. tensor-free 异常定义放在 `pssolver.core.errors`；`pssolver.errors` 保留为公共兼容导出入口，避免底层 linear-solver 反向依赖架构层。

## 验证结果

- 直接 RC6-4 合同测试：4 passed。
- 受影响的 error、workflow、functional 与 pressure 测试：135 passed。
- 原 A7-03 不稳定输入独立复现矩阵：18/18 个 Plane/Channel 变体全部 fail closed；异常全部属于 `SimulationExecutionError`；已启动 workflow 的 metadata 全部为 `failed`。
- 历史 successor 审计：121 passed；最终级联复核 20 passed。
- 完整 CPU suite：3912 passed，1 skipped，7 deselected，8 subtests passed。
- 唯一 skip：可选 `nematics3d` 依赖不可用；没有新增 skip、xfail 或 failure。
- `git diff --check`：PASS。

## 历史证据处理

旧资格 JSON 与旧发布记录均未改写。因本阶段产品文件和审计测试的内容哈希发生变化，RC6-4 记录通过 `historical_successor_sources` 显式登记前驱哈希、当前哈希和原因；旧测试只增加对 RC6-4 forward-only successor 的解析。

机器可读记录：`notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json`。

## 后续边界

RC6-4 完成后可进入 RC6-5 的合同核对与本地实现；RC6-4 本身不授权自动合并、默认值提升、push 或 H100 submission。
