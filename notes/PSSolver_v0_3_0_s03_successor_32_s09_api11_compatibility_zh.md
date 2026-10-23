# S-09 API 1.1 历史 checkpoint 兼容修复

S-09 的公开 `execution_scope()` 使 functional construction API 从 1.0
提升到 1.1。完整 CPU 门禁发现，显式迁移旧 checkpoint 时错误地把当前
1.1 literal 用于校验历史 1.0 bridge 和 migration provenance。

`dc12dc6fbf984fb1a98e1de672903e51750742f3` 修正这一边界：当前
runtime metadata 可以投影到冻结的 1.0 identity schema，但迁移校验和迁移记录
继续使用源 checkpoint 的 1.0 literal；未知的未来协议仍失败关闭。历史 P9、rc5
和 rc6 记录没有被改写，测试改为分别核对历史 literal 与当前兼容 reader。

修复不改变 timestep、压力求解热路径、checkpoint schema、provider 的科学身份、
性能阈值或生产默认值。S-09 H100 候选因此从 `2573dec` 前移到包含本修复的
`dc12dc6`；此前 CPU 性能遥测保留为热路径不变的参考证据。
