# PSSolver v0.2.0rc6 released-checkpoint compatibility 与 tamper CPU 关闭

## 结论

本门禁通过：

`PASS_RC6_RELEASED_CHECKPOINT_COMPATIBILITY_AND_TAMPER_CPU`

受验证对象仍是冻结 source commit
`ea801ec08f5b2465661334e1c126133e55b68e96` 构建的 installed wheel；本提交只增加
qualification runner、证据记录和记录测试，不改变已验证的 runtime。

## 已发布 checkpoint

- rc1–rc4 installed-wheel 测试：20 passed，0 failed，0 skipped；
- 冻结处置案例：13 个；
- 仓库 fixture 与最初只读 post-release 归档逐树交叉校验：10/10 PASS；
- RC5 历史记录及其显式迁移、精确升级、直接读取和 fail-closed Nyquist
  拒绝语义均未改写。

## RC6 当前 checkpoint

以下四类 CPU→CPU 单元全部通过：

1. Periodic functional；
2. Channel functional fixed-iteration；
3. Channel functional tolerance；
4. legacy Channel。

四类都满足：

- 正常恢复与 continuous/split restart replay 精确一致；
- raw-byte payload 篡改被 integrity/checksum guard 拒绝；
- identity 篡改被 provenance/compatibility guard 拒绝；
- 两类拒绝均发生在 target mutation 之前；
- source checkpoint 未修改；
- 状态和观测有限；
- 没有 runtime fallback。

## 边界

RC6 没有改变 checkpoint schema 或 functional API 1.0。本轮没有 push、H100
submission、tag、Release 或 PyPI 操作，没有修改 PSSolver-Control、nematics3d 或生产默认值。

下一道门禁是 provider identity impact review；在该审查结束前，RC6 仍不具备 push 或 H100
提交资格。
