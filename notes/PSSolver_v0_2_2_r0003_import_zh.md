# PSSolver v0.2.2 候选发布验收 R-0003 导入记录

日期：2026-10-21

分类：`PASS_R0003_V0_2_2_CANDIDATE_ACCEPTANCE_IMPORT`

本提交只导入 Claude 对 v0.2.2 候选发布源与本地产物的独立只读验收结果，
不修改运行时、问题登记表、历史记录、tag、Release、资产、PSSolver-Control
或 nematics3d，也不运行 HPCC 或 H100。

## 导入结论

- R-0003 classification：`PASS`；
- release blockers：0；
- `V022-C1/C2/C3/C5`：`PASS`；
- `V022-C4`：`COVERED_BY_EVIDENCE`；
- 完整 CPU suite：4191 passed、7 deselected、8 subtests、0 failed/skipped；
- installed-wheel 门禁：PASS；
- H100：不需要、未运行；
- registry updates：0；
- new findings：0。

R-0003 确认候选运行时差异仅为已经 R-0002 验证的 V-17 一行修复与版本号提升；
checkpoint 树与 v0.2.1 逐文件一致，Periodic/Channel provider identity 不变，
Control provider pins 可接受。v0.2.2 可以进入另行授权的 push、tag 和 GitHub Release
协调阶段，但本导入不授予任何远端操作权限。

Claude 只读归档：

`/home/fansenwei/pssolver_v0_2_2_claude_candidate_acceptance_20261020_v1`

其 `checksums.sha256` 包含 32515 项，manifest SHA-256 为：

`a8f649d5d6a4354028a87ff690dc1325d5306ceca3395332ff1db9ef0813a6c6`

同步结果 `R-0003.result.json` SHA-256 为：

`0f68e73a6d3cd130d44a2f41dfca9867702570049ec2c5f670a70dc13a03c91a`

问题登记表保持 28 条且字节不变；下一轮同步编号从 `R-0004` 开始。

push、tag、Release、PyPI、HPCC 与 H100 仍需用户逐项授权。
