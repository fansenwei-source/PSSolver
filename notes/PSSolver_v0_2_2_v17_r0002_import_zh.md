# PSSolver V-17 独立复核 R-0002 导入记录

日期：2026-10-20

分类：`PASS_R0002_V17_INDEPENDENT_REVIEW_IMPORT`

本提交只导入 Claude 对 V-17 的独立复核结果，不修改运行时、历史记录、tag、Release、
资产、PSSolver-Control 或 nematics3d，也不运行 H100。

## 导入结果

- `V-17`：从 `open` 更新为 `verified`；
- `fixed_in`：`611d167d1b2c2d634222f608f4c9cd4dc671636d`；
- `verified_by`：`R-0002`；
- 新增 `V-22`：事务中途改指父 symlink 后，旧目标目录可能留下 staging；
- 新增 `V-23`：不可搜索 checkpoint 父目录可能暴露原始 `PermissionError`；
- `A2-11`、`B8/S2-07`、`R1-R9` 均绑定已确认决定 `D-0007`。

登记表由 26 条增加到 28 条；新问题均为低严重度、目标 `0.2.x`、影响 Control
公开 checkpoint 路径但不由当前 Control 研究执行。V-22 与 V-23 均不推翻 V-17 的 PASS。

Claude 只读归档：

`/home/fansenwei/pssolver_v0_2_2_v17_claude_review_20261020_v1`

其 `checksums.sha256` 包含 32253 项，manifest SHA-256 为：

`418f298d2dfcc81f99c0d3574f7c9c5d153a6c4483cc6f1732c050365980b9a8`

本地同步结果文件为 `R-0002.result.json`，SHA-256 为：

`8837d79a34997a762cca26f967df3396ea87b5208afc39a744fe2c6f9f32c93d`

push、main/maint 快进、dev/0.3 cherry-pick、tag、Release 与 H100 均仍未获授权。
