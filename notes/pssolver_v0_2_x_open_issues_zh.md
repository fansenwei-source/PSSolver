# PSSolver v0.2.x 未解决问题统一登记表

日期：2026-10-23

本登记表是可持续维护的 evidence-only successor，自 `D-0010` / `S-01` 起以
`dev/0.3` 为主副本。历史发布记录保持逐字节不变；
`PSSolver_v0_2_issue_tracking_zh.pdf` 仍是持续更新的说明材料，但不作为固定哈希输入。

当前共登记 39 条：

- `open`：1；
- `registered`：1；
- `claimed_fixed`：0；
- `verified`：33；
- `deferred`：2；
- `accepted`：1；
- `coverage_gap`：1。

## 当前优先项

1. `V-17` 已由提交 `611d167d1b2c2d634222f608f4c9cd4dc671636d` 修复，并经
   `R-0002` 独立验证为 `verified`。验证覆盖真实跨文件系统父目录 symlink、Periodic/Channel
   functional export、forced fallback、installed wheel、真跨文件系统拒绝与完整 CPU suite。
2. `NEW-14`：在 `0.3 GPU verification` 中核实 TF32 policy scope 是否覆盖 backward；
   目前只登记，不宣称存在数值错误。
3. S-04 的六项 checkpoint 健壮性问题已经完成：V-22、V-16、V-13、V-14 与 C2-17
   由提交 `468e067` 修复，V-25 由提交 `44a6fc3` 修复；本地复核和 HPCC 真实 NFS
   门禁均通过，并由 `R-0012` 独立验证为 `verified`。
4. `A2-21/S2-12` 已由 S-05 运行时提交
   `65ef1080cb0215131efe90143863de7014fe4b5b`（包含 `af7c121…` 的实现）修复，
   并由 `R-0013` 独立验证为 `verified`。
   新写出的 Plane 3、legacy Channel 3、finite-Q 2 与 experimental shadow 2 格式均对完整元数据
   加 SHA-256 seal；旧 v0.2 格式保持只读。该摘要不是 MAC 或签名，蓄意重算摘要不在 v0.3
   威胁模型内。
5. `R-0016` 已完成 S-06 的合并独立复核：`C1-05`、`C2-18`、`V-27` 的修复身份为
   `bbb7077`，`S2-17`、`V-28`、`V-29` 的修复身份为 `f6da20f`；六项均为 `verified`。
6. `S2-17` 与 `V-28` 由 `f6da20f` 按修订后的方案 B 修复：
   float32 fixed-work PCG 保留下限内递推残差最小的状态，仅在实际回滚时严格要求返回状态真实残差
   不高于精度底。指定的 2160 次压力扫描 `fail_closed=0`，且没有真实残差高于下限的可接受结果；
   R-0016 已独立复核通过。
7. `V-29`（low）也由 `f6da20f` 修复：正常完成且未发生回滚时保留旧有接受语义，
   真实残差可使用不超过两个精度底的有限余量；但在使用余量前仍优先采用满足严格后条件的已存
   下限候选。指定的 360 次首个下限交叉扫描 `fail_closed=0`，精确复现案例通过；R-0016 已独立
   复核通过。
8. S-07 的六项 `A1-07`、`S1-15/NEW-13`、`S1-04`、`OBS-4`、`B30`、`V-26`
   已由实现提交 `b7cf4ca02095e8b41ed720809880628d2375cfbb` 修复，并由
   `R-0018` 独立验证为 `verified`。`R-0017` 曾对这六项逐项复核通过，但发现候选 wheel 排除了
   `pssolver.control` 后，保留的 `pssolver.channel` 仍导入该包，导致两条 Channel
   installed-wheel runtime 均无法构建。
9. 上述候选回归 `V-30`（high）已由运行时提交 `8aceaab` 修复，并由
   `R-0018` 独立验证为 `verified`：共享 active-force 实现已移入实际打包模块，同时保留源码
   兼容转发；wheel 全模块导入、两条 Channel runtime 构建、四条公开应用真实 dry-run、完整
   CPU suite 与历史 successor 门禁均通过。
10. S-08 已由实现提交 `8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851` 补入真实历史
    CPU fixture：rc1/rc2 Channel functional 奇数周期轴 checkpoint 以及 rc1 Plane lifting
    checkpoint。rc2 奇数轴可精确续算；rc1 因 B2 之前的不可压缩动力学只能安全执行
    `state_only` 迁移，不能虚假声明精确续算；真实 lifted checkpoint 在读取 tensor payload 前
    明确拒绝，且不重开 lifting。`A2-25` 与 `N5` 暂为 `claimed_fixed`，等待 `R-0019` 独立复核。
11. `R-0019` 独立复核确认 fixture 可逐字节重新生成、rc1 的 state-only 改判正确、N5
    真实负向门正确，但要求返工两项低风险候选问题：`V-31` 为 continuation oracle 的构建指纹
    与跨构建容差合同不完整；`V-32` 为旧版本误标 rc1 奇数轴迁移被拒绝时缺少专门、可操作的
    错误说明。两项均登记到 S-08，`A2-25` 与 `N5` 保持 `claimed_fixed`。
12. `V-31` 与 `V-32` 已由返工提交
    `e2343d19301de146ca632a1a12f7c9659e598f73` 修复并推进为 `claimed_fixed`：fixture
    记录完整数值构建指纹，rc2 同构建逐位、跨构建使用 `8·eps`，rc1 差异必须至少高于舍入
    容差一千倍；旧版误标的 rc1 奇数轴 checkpoint 会收到要求从原始 rc1 checkpoint 重新执行
    state-only migration 的专门错误。`R-0020` 确认全部数值行为通过，但发现记录层仍需返工：
    `V-33`（low，S-10）登记 CUDA 可见性会改变数值构建指纹、使 CPU 门禁未覆盖同构建逐位分支；
    `V-34`（low，S-08）登记历史记录中由短哈希错误手工补全出的完整提交号。四项 S-08
    claim 保持 `claimed_fixed`，等待记录勘误后的 `R-0021` 复核。
13. `R-0021` 已独立验证 S-08 的数值结论和纯记录勘误：`A2-25`、`N5` 在
    `8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851` 修复，`V-31`、`V-32` 在
    `e2343d19301de146ca632a1a12f7c9659e598f73` 修复，`V-34` 在
    `dea728f1a926284420e6b1a17edb56eb56ea3db1` 修复，五项均为 `verified` 且
    `verified_by=R-0021`。`V-33` 继续保持 `registered`，在 S-10 处理。S-08 完成。

## Control 路径

`C1-05`、`C1-10`、`C1-11`、`C2-18`、`C1-08`、`NEW-14`、`A3-18`
以及 `V-17`、`V-22`、`V-23`、`V-16`、`V-13`、`V-14`、`V-25`、`V-21`、`OBS-4` 会触及 Control
依赖或暴露的路径。当前 Control 研究没有执行这些已登记的 checkpoint 限制路径，所有条目的
`exercised_by_current_control_research` 均为 false。

## 0.3 路线图

S-07 的 6 个 low/info 条目与候选回归 `V-30` 均已由 `R-0018` 独立验证为
`verified`，S-07 完成；S-08 的 `A2-25`、`N5`、`V-31`、`V-32`、`V-34`
均已由 `R-0021` 独立验证，S-08 完成；`V-33` 留待 S-10。
`A2-21/S2-12` 已完成 `S-05` 独立复核。`V-27` 归入 `S-06`，`V-26` 与
`B30` 归入文档批次 `S-07`。
`V-17` 已验证身份不变，不重新打开。路线图顺序以同步仓库中用户确认的
`D-0010` 与 `plan/roadmap.json` 为准。

## 延期与覆盖缺口

- lifting：`B8/S2-07`，按已确认决定 `D-0007` 在重新开放 lifting 前处理；
- 架构整理：`R1-R9`，按 `D-0007` 推迟至 0.3 之后；
- `A2-11`：按 `D-0007` 接受为设计限制；
- 剩余覆盖缺口：`A3-18`，在 0.3.0 完整验收前补齐。

机器可读的逐项标题、严重度、来源、版本影响、Control 影响、状态、目标版本和说明均在
`notes/pssolver_v0_2_x_open_issues.json` 中。路线图 S-08 已完成，下一步进入 S-09
GPU / H100 集中资格任务；任何 GPU、Slurm 或 H100 执行仍需用户单独授权。
