# PSSolver v0.2.x 未解决问题统一登记表

日期：2026-10-21

本登记表是可持续维护的 evidence-only successor，自 `D-0010` / `S-01` 起以
`dev/0.3` 为主副本。历史发布记录保持逐字节不变；
`PSSolver_v0_2_issue_tracking_zh.pdf` 仍是持续更新的说明材料，但不作为固定哈希输入。

当前共登记 30 条：

- `open`：1；
- `registered`：9；
- `claimed_fixed`：6；
- `verified`：8；
- `deferred`：2；
- `accepted`：1；
- `coverage_gap`：3。

## 当前优先项

1. `V-17` 已由提交 `611d167d1b2c2d634222f608f4c9cd4dc671636d` 修复，并经
   `R-0002` 独立验证为 `verified`。验证覆盖真实跨文件系统父目录 symlink、Periodic/Channel
   functional export、forced fallback、installed wheel、真跨文件系统拒绝与完整 CPU suite。
2. `NEW-14`：在 `0.3 GPU verification` 中核实 TF32 policy scope 是否覆盖 backward；
   目前只登记，不宣称存在数值错误。
3. `V-25`：`R-0010` 在 S-04 预审中发现的 medium-low 回退；明确的 NFS 占位目录
   `mkdir` 失败被误判为模糊状态并保留完整 staging。本条已纳入 S-04 返工；此前五项
   S-04 修复仍保持 `claimed_fixed`；V-25 已由提交 `44a6fc3` 返工并同样标记为
   `claimed_fixed`，六项一起等待 R-0011 短复核和真实 NFS 复测。

## Control 路径

`C1-05`、`C1-10`、`C1-11`、`C2-18`、`C1-08`、`NEW-14`、`A3-18`
以及 `V-17`、`V-22`、`V-23`、`V-16`、`V-13`、`V-14`、`V-25`、`V-21`、`OBS-4` 会触及 Control
依赖或暴露的路径。当前 Control 研究没有执行这些已登记的 checkpoint 限制路径，所有条目的
`exercised_by_current_control_research` 均为 false。

## 0.3 路线图

9 个仍为 `registered` 的 low/info 或明确指定条目统一以 `0.3` 为目标，
`A2-21/S2-12` 归入设计先行的 `S-05`，`B30` 归入文档批次 `S-07`。
`V-17` 已验证身份不变，不重新打开。路线图顺序以同步仓库中用户确认的
`D-0010` 与 `plan/roadmap.json` 为准。

## 延期与覆盖缺口

- lifting：`B8/S2-07`，按已确认决定 `D-0007` 在重新开放 lifting 前处理；
- 架构整理：`R1-R9`，按 `D-0007` 推迟至 0.3 之后；
- `A2-11`：按 `D-0007` 接受为设计限制；
- 覆盖缺口：`A2-25`、`A3-18`、`N5`，在 0.3.0 完整验收前补齐。

机器可读的逐项标题、严重度、来源、版本影响、Control 影响、状态、目标版本和说明均在
`notes/pssolver_v0_2_x_open_issues.json` 中。本次只登记，不实施任何修复。
