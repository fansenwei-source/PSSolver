# PSSolver dev/0.3 N-1 与 V-17 的 R-0005 独立复核导入

日期：2026-10-21

分类：`PASS_R0005_DEV03_N1_V17_INDEPENDENT_REVIEW_IMPORT`

本提交只登记 Claude 对本地 `dev/0.3` 候选
`f45518ca34b7901d64c5ee43e2e739067fae8451` 的独立只读复核结果。
它不修改运行时、打包输入、问题登记表、历史记录、tag、Release、资产、
PSSolver-Control 或 nematics3d，也不运行 HPCC、GPU 或 H100。

## 独立结论

- R-0005 classification：`PASS`；
- release blockers：0；
- `V03-CKPT-1` 至 `V03-CKPT-5`：全部 `PASS`；
- `V03-PLAN-1`：`PASS`；
- registry updates：0；
- new findings：0；
- `dev/0.3` 可在单独授权后进行非强制快进 push。

Claude 确认 `pssolver/io/checkpoint.py` 与 v0.2.2 发布源逐字节相同，
N-1 与 V-17 行为均通过独立负向和正向对照，版本保持
`0.3.0.dev0`，checkpoint schema、provider identity、方程、数值算法和
production defaults 均未改变。

Claude 的完整 CPU 环境具备 Inductor，结果为 4239 passed、
7 个精确 CUDA-only deselect、8 个 subtests、0 failed/skipped。
这也确认 Codex 环境里的 6 个 skip 只是既有 Inductor 能力条件，而非代码失败。

独立归档：

`/home/fansenwei/pssolver_dev03_claude_review_20261021_v1`

归档含 `COMPLETE`，`checksums.sha256` 共 2984 项，manifest SHA-256：

`2a9509ea1b096f656130cbf863b295cfea58c9c65f1e2f460954261021825a07`

同步结果 `R-0005.result.json` SHA-256：

`5c5f01b5904fb6f80be678d819bdebd28e67d2014fe84cbbd083f07b2d1a76a8`

本次不改登记表。下次登记表 successor 会把剩余 v0.2.x low/info 条目及
NEW-14 的 `target_release` 与 0.3 计划统一；以后 request 的
`created_at` 使用创建时的系统实时时间。

任何 push、main/maint 快进、tag、Release、PyPI、HPCC、GPU 或 H100
操作仍需用户单独授权。下一轮同步编号从 `R-0006` 开始。
