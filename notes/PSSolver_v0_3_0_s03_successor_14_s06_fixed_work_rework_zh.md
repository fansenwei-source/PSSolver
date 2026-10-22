# S-06 fixed-work 压力返工候选

实现提交 `4c3a2d8bca191b6aa361c37920826e63582a4774` 按用户选择的方案 B
同时返工 `S2-17` 与 `V-28`：固定迭代过程中保留最后一个递推残差不高于精度下限的状态；
随后发生回升、算子或预条件 breakdown、非有限状态时回滚，并以
`fixed_work_roundoff_saturation` 结束。接受前必须根据实际返回压力重算 represented residual；
只有它不高于未改变的八倍 dtype epsilon 下限才可接受，否则失败关闭。

回归矩阵使用 R-0014 的两组精确反例、200/250/300/400 四档预算，并覆盖 production、
functional transpose、implicit primal 和 unrolled oracle。没有采用一触底便停止、会改变普通结果的
方案 A。`C1-05`、`C2-18`、`V-27` 仍保持第一版 S-06 的修复身份；五项统一等待 R-0015。

本候选不授权 push、HPCC、GPU 或 H100，不修改压力下限、checkpoint schema、provider identity
或生产默认值。
