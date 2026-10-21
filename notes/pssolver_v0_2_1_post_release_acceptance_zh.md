# PSSolver v0.2.1 独立发布后验收登记

日期：2026-10-20

分类：`PASS_V0_2_1_INDEPENDENT_POST_RELEASE_ACCEPTANCE_WITH_DOCUMENTED_LIMITATIONS`

本文件是只追加的 evidence-only successor，不是发布源码，也不改写 v0.2.1
tag、GitHub Release、资产或历史资格记录。Claude 的只读验收归档位于：

`/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1`

归档的 `checksums.sha256` 含 10102 项，全部验证通过；manifest SHA-256 为
`332543391e608dd9dc778d3b163d025e0d42c36ed2b2290b5097b73cc3190b70`。
`COMPLETE` 存在。唯一未进入 manifest 的文件是验收程序故意设置为不可读的故障注入
payload，其排除已经由 `COMPLETE` 明确记录。

## 结论

- blocker：0；相对 v0.2.0 的回归：0；
- clean tag worktree CPU：4135 passed、7 deselected、0 failed；
- 本机 N-1 合同 23 条：17 PASS、5 项已记录限制、1 项由 HPCC 真机补测；
- HPCC `/scratch1` 与 `$HOME` 的真实 NFS 上，v0.2.0 fail-closed，v0.2.1
  可正确发布；重复目标拒绝、并发唯一赢家、restart 逐位一致、NFS 与 `/tmp`
  checkpoint 逐字节一致；
- 既有 NFS CPU 与 H100 归档交叉核对一致，未重新提交 H100；
- Control provider recheck 逐位一致，identity 与 pin 一致，当前研究不执行 durable
  checkpoint 路径。

## 已记录但不阻塞 v0.2.1 的限制

`V-17`、`V-16`、`V-13`、`V-14`、`V-21` 与 `OBS-4` 已迁入
`notes/pssolver_v0_2_x_open_issues.json`。其中 V-17 是下一次 v0.2.x 维护的优先候选；
本登记不实施任何修复。

本提交不修改 `pssolver/`、PSSolver-Control、nematics3d、生产默认值、公共 API、
checkpoint schema、tag、Release 或资产，也不需要 H100 重新资格化。
