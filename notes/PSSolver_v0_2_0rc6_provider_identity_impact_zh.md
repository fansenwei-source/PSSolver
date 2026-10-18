# PSSolver v0.2.0rc6 provider identity 影响审查

## 结论

本门禁通过：

`PASS_RC6_PROVIDER_IDENTITY_UNCHANGED_NO_CONTROL_IDENTITY_MIGRATION_REQUIRED`

RC6 冻结 installed wheel（source `ea801ec08f5b2465661334e1c126133e55b68e96`）
与已发布 RC5 installed wheel（source
`674a8121779be72c95aa5dc9e2babca0ce7b93a9`）在三类 canonical Consumer
fixture 上的完整 functional Provider identity 完全一致。因此 RC6 不需要新的
PSSolver-Control provider-identity 迁移，也不需要更新 identity pin。

## 比较方法

比较使用 RC5 最终 G-1 资格冻结的 Consumer source
`c14e6b56c88081e7f9a5642e65ffd722eec7445b`，而不是早期诊断用的小网格 digest。
同一 fixture 分别驱动 RC5 和 RC6 installed wheel；每一类又在两个不同输出根目录中独立
构建，以验证 identity 不携带输出路径。

每个 runtime 都保存完整 `identity().to_metadata()`，并独立重算 compact canonical JSON
SHA-256，与 `identity().canonical_sha256()` 比较。重新构建的 RC5 完整 metadata 还与
历史 installed-wheel 证据逐字段一致。

## 结果

| fixture | RC5 / RC6 canonical SHA-256 | 变更路径 |
|---|---|---:|
| Periodic canonical | `d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d` | 0 |
| Channel fixed iteration | `b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905` | 0 |
| Channel tolerance | `b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908` | 0 |

三类都满足完整 metadata 相等、digest 相等、RC5/RC6 路径无关；functional API 仍为
`1.0`。本审查没有执行科学 timestep。

## 与 Control 后续工作的边界

“无需 identity 迁移”不等于“Control 已全部完成 RC6 资格验证”。已有独立分支
`fix/pssolver-v0.2.0rc6-guard-order` 已完成 stale-pin guard-order 的 CPU 回归关闭，
但尚未以 RC6 冻结 Provider wheel 完成 installed-wheel CPU 审查，也没有 push。

因此下一道门禁是该 Control source 与 RC6 冻结 wheel 的 installed-wheel CPU 审查。
此 identity review 不需要也未授权 H100。

本轮没有修改 PSSolver runtime、PSSolver-Control、nematics3d、生产默认值、tag、Release
或 PyPI；没有 push，也没有 H100 submission。
