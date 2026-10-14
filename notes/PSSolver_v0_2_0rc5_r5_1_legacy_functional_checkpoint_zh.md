# PSSolver v0.2.0rc5 R5-1 旧版 functional checkpoint 兼容修复

本记录关闭 `B10` 的实现缺陷，但 `rc5_must` 批次仍有其他事项，因此暂不推进发布资格。

## 根因

rc4 reader 使用当前 runtime identity 推导旧 bridge identity，并把真实 rc1 错误建模为 `0.1-provisional`。真实 rc1、rc2、rc3 checkpoint 的 `api_version` 均为 `1.0`；Periodic 与 Channel 还分别拥有按发布世代固定的字段集合。

## 修复

- 新增按 `rc1`、`rc2`、`rc3` 和 Periodic/Channel family 区分的冻结 canonicalizer；
- canonicalizer 显式枚举历史 `functional_runtime` 字段，当前版本以后新增的字段不会改写已发布 digest；
- rc2/rc3 Periodic v2 可直接读取并精确升级；
- rc1 Periodic 保持显式、非原地的 state-only 重投影合同；
- rc1/rc2 Channel 偶数周期轴保持 state-only migration，rc3 Channel 为 exact upgrade；
- 所有迁移保持源 checkpoint 逐字节不变，不允许 silent fallback。

## 真实 fixture

来自 rc4 发布后只读验证目录的六套 functional checkpoint 已逐字节纳入：

`tests/fixtures/functional_checkpoint_releases_v1`

`manifest.json` 绑定原始来源、rc1–rc3 release commit、checkpoint tree SHA-256、runtime identity、production identity、state-layout identity 和构造 snapshot SHA-256。

## 当前边界

- 不修改 `v0.2.0rc4` tag、Release 或历史 closure；
- 不修改 PSSolver-Control 或 `nematics3d`；
- 不修改 functional API 版本或生产默认值；
- 本项不需要 H100；
- 尚未 push；
- 完整 source 与 installed-wheel CPU 结果完成后再把记录状态更新为最终 PASS。
