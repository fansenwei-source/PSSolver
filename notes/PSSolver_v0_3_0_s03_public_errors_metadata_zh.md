# PSSolver v0.3 S-03：公共异常与失败元数据合同

状态：本地候选完成，等待 R-0008 独立验收。

实现提交为 `a654e1c1d97fa364ac32b6ccbf93070056066d21`，直接基于
`19dcd0be83c86e8cace6abff714d3a9f55234c0b`。本阶段处理：

- C1-10：补全 `FunctionalConvergenceError` 的公共执行异常元数据；
- C1-11：已启动 workflow 遇到任意中断时持久化 `failed` 状态，并把非有限详情转换为严格 JSON token；
- A2-24/C1-15：公开 restart 拒绝统一为 `SimulationRestartError`；
- C1-08：无效参数在 compile 阶段结构化拒绝，缺失、dtype 错误或非有限 snapshot 统一为 `SimulationInputError`；
- V-23：不可搜索 checkpoint 父目录映射为 `CheckpointIOError`；
- V-21：Periodic functional reader 的不可读占位映射为 `FunctionalCheckpointIOError`。

同时在 G-04 登记：标准 Plane、Periodic、Channel 算例必须与 v0.2.2 做数值不退化比较；代码路径未变处要求逐位相同，变化处只使用预先登记的容差。

本阶段没有改变方程、数值算法、checkpoint schema、provider identity、生产默认值或测试阈值；没有增加 skip/xfail；没有修改 PSSolver-Control 或 nematics3d；没有运行或授权 HPCC、GPU、H100、push、tag 或 Release。

旧 evidence 记录保持不变。机器可读记录中的 `historical_successor_sources` 只在前驱哈希精确匹配时推进到 S-03 当前哈希。
