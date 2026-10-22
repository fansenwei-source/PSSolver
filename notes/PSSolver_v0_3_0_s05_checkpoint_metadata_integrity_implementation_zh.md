# PSSolver v0.3.0 S-05 checkpoint 元数据完整性实现

S-05 的运行时实现提交为
`af7c1217b4599f7e870b42a9c0fc92f5b3c90c61`，父提交为已确认设计
`26216537ee38c2258e06f32ce9b095d17a9dbc40`。用户对 S05-Q01 至 S05-Q05
全部选择 A。

新写出的 Plane format 3、legacy Channel format 3、finite-Q format 2 和
experimental shadow format 2 会在所有 tensor record、进度、restart、identity 与迁移字段
确定后，对完整 metadata 计算 canonical SHA-256。读取新格式时先校验摘要，再读取 tensor；旧版
格式继续走精确、只读的版本分支，并以 `unsealed_legacy` 暴露其完整性状态。

四类升级均为 source-to-new-target。新 checkpoint 的受摘要保护迁移记录包含
`source_integrity_status=unsealed_legacy`、源目录 tree SHA-256、
`source_historical_integrity_authenticated=false` 和
`source_checkpoint_mutated=false`；公共 writer 会拒绝缺少这些字段的升级声明。因此升级只证明
本次看到并转换了哪些源字节，不声称未签名源在历史上从未被修改。

已发布 v0.2.2 wheel 对四类新格式全部失败关闭，并分别给出 Plane、Channel、finite-Q 和
shadow 的明确格式或 schema 不兼容错误；它们不能被 v0.2.x 续算。冻结的旧 v0.2 checkpoint
fixture 保持读取与一步续算兼容。

这里的 SHA-256 是完整性摘要，不是 MAC 或数字签名。它能发现误改、部分修改、损坏，以及没有
同步重算摘要的协调修改；不能阻止有意同时重写 metadata 与摘要的攻击者。后者不在 v0.3
范围内。

本地 CPU 与 installed-wheel 证据将随 R-0013 一并封存后交 Claude 独立复核。本记录不授权
push、tag、Release、HPCC、GPU 或 H100。
