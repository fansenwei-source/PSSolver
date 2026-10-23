# S-09 checkpoint 兼容返工候选

运行时提交 `85ba3aef75548d549d9db453ca4a5ab604b17a9d` 修复 R-0022 登记的
`V-35` 与 `V-36`，两项现为 `claimed_fixed`，仍须由 `R-0023` 独立复核。

- 旧 rc1/rc2/rc3 migration 与 Periodic v2 bridge 固定要求 API `1.0`；`1.1`、
  `0.1-provisional` 和未来版本均在 tensor payload 读取前拒绝。迁移记录写固定的发布版本，
  不再复制可由输入控制的标签。
- 来自 v0.2.2 正式运行时的 rc1 偶数轴 Channel state-only migration fixture 保持完整
  tree hash。旧标签与 v0.3 展开的标签表达相同保守语义，允许读取并有限续算，但不宣称
  trajectory equivalent。
- checkpoint 格式、tensor payload、provider identity、模型方程、阈值、默认值均未改变；
  没有 push、HPCC、GPU 或 H100 操作。
