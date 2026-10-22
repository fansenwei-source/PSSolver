# PSSolver v0.3.0 S-06 数值与可序列化候选

日期：2026-10-22

运行时资格源为 `bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a`。本地候选覆盖
`C1-05`、`C2-18`、`S2-17` 和 `V-27`，四项在独立 `R-0014`
复核前均只登记为 `claimed_fixed`。

- tolerance-mode 的公共下限仍为 8 个 real-dtype epsilon；达到递归残差门限后，
  必须用实际返回压力重新计算残差。若仍不满足，便从该残差重新启动 PCG recurrence。
- 普通 fixed-work 路径保持既有计算顺序；只有当前残差已经达到支持的精度底、下一次
  更新将 breakdown 或变成非有限值时，才以
  `fixed_work_roundoff_saturation` 明确结束。精度底以上的 breakdown 继续失败关闭。
- `ChannelPressureConvergenceError` 及双继承的
  `FunctionalConvergenceError` 经 pickle 往返后保持精确类型、args、机器元数据、
  diagnostics 和 functional operation。
- 两条 S-04 导入测试核对记录里的字面 `sys.path`，不再依赖当前 checkout 路径。

本阶段不改变 checkpoint schema、provider identity、functional API 版本、生产默认值，
不增加 skip/xfail，也不执行 push、HPCC、GPU 或 H100。
