# PSSolver dev/0.3 S-01 的 R-0006 独立复核导入

日期：2026-10-21

分类：`PASS_R0006_S01_REGISTRY_MIGRATION_REVIEW_IMPORT`

本记录把同步结果 `R-0006` 以纯证据方式导入 `dev/0.3`。独立复核确认
S-01 已通过：登记表主副本迁到 `dev/0.3`，20 个 `registered` 条目均指向
0.3，`NEW-14` 指向 `0.3 GPU verification`，`A2-21/S2-12` 指向
0.3，`B30` 归入 S-07，且 V-17 的既有 verified 证据保持不变。

## 身份与边界

- R-0006 结果提交：`4be9c56c3ee46d44eda3bb5388e5c458a575f54f`；
- 被复核候选：`2b3d6c49dae53bdec19c96c39cfbaaaf71692262`；
- 远端基线：`7301bd4bf0b49b27e1c07d248ea3ca3f483dc147`；
- R-0006 classification：`PASS`；
- registry updates：0；new findings：0；
- 独立完整 CPU：4252 passed、7 deselected、8 subtests passed，0 failed、
  0 skipped；
- 本次导入不改登记表、运行时、打包、数值、checkpoint schema、provider
  identity 或生产默认值，也不授权任何远端、HPCC、GPU 或 H100 操作。

## 测试环境合同

本次导入测试只从 D-0008 明确列出的基线环境
`/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1/a2/venv_wheel`
用 `cp -al` 创建硬链接克隆。环境记录保存了精确克隆来源、Python 可执行文件、
`sys.prefix`、`sys.path` 和 pytest 导入路径；测试结束后删除克隆。

今后的测试环境只允许克隆 D-0008 列出的基线 venv，并且每份环境记录都必须至少
写明 `baseline_venv_source` 和实际 `sys.path`。R-0006 提到的原 S-01 Codex
环境来源问题只作为信息保留；它不改变独立复核的 PASS 结论。

## 后续

按 D-0009，可在同步检查为 0 error、0 pending 后，请用户授权将
`dev/0.3` 非强制快进推送。推送完成后的路线图下一步为 S-02；下一同步编号为
R-0007。
