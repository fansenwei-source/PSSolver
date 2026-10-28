# PSSolver v0.2.x 未解决问题统一登记表

日期：2026-10-27

本登记表是可持续维护的 evidence-only successor，自 `D-0010` / `S-01` 起以
`dev/0.3` 为主副本。历史发布记录保持逐字节不变；
`PSSolver_v0_2_issue_tracking_zh.pdf` 仍是持续更新的说明材料，但不作为固定哈希输入。

当前共登记 50 条：

- `open`：0；
- `registered`：1；
- `claimed_fixed`：1；
- `verified`：45；
- `deferred`：2；
- `accepted`：2；
- `coverage_gap`：0。

## 当前优先项

1. `V-17` 已由提交 `611d167d1b2c2d634222f608f4c9cd4dc671636d` 修复，并经
   `R-0002` 独立验证为 `verified`。验证覆盖真实跨文件系统父目录 symlink、Periodic/Channel
   functional export、forced fallback、installed wheel、真跨文件系统拒绝与完整 CPU suite。
2. `NEW-14` 已由 `R-0027` 在 H100 上验证：Periodic / Channel × 六种 caller policy
   的 forward 与 backward hook 均观察到作用域内策略，成功、异常及嵌套作用域后都精确恢复，
   作用域外保持调用者策略；修复身份为 `85ba3ae…`。
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
14. `R-0022` 的 S-09 H100 前独立复核要求两项小范围返工：`V-35` 固定旧格式迁移与
    v2 bridge 的 API 版本为发布时的 `1.0`，并在读取张量前拒绝伪造或未来版本；`V-36`
    兼容 v0.2.x 对 rc1 偶数轴 Channel state-only 迁移使用的历史标签。两项登记为
    `registered`，均在 S-09 唯一 H100 作业前完成；A3-18 与 NEW-14 状态不变。
15. `R-0023` 已独立验证 `V-35`、`V-36` 的运行时实现，二者在
    `85ba3aef75548d549d9db453ca4a5ab604b17a9d` 推进为 `claimed_fixed`。复核同时发现
    `V-37`：V-36 的两个 `.npy` payload 被全局 ignore 规则排除，导致干净 detached worktree
    各有一个 fixture 测试失败。V-37 登记在 S-09，须提交精确 payload、增加 Git 跟踪守卫并在
    干净 worktree 重跑两套完整测试；A3-18 与 NEW-14 状态不变。
16. `V-37` 已由 `87cf34cf4e17096a701f5cc5bc81076d14a9d237` 修复：两个 payload
    已按 manifest 的精确哈希纳入 Git，并新增 manifest 全文件跟踪守卫；Nematics3D 与 pip-torch
    两套完整 suite 均已从干净 detached worktree 通过。该项推进为 `claimed_fixed`，等待
    `R-0024` 独立复核；`V-35`、`V-36` 仍保持 `claimed_fixed`，A3-18 与 NEW-14 不变。
17. `R-0024` 已独立验证 `V-35`、`V-36`（修复身份 `85ba3ae…`）和 `V-37`
    （修复身份 `87cf34c…`），三项均推进为 `verified`、`verified_by=R-0024`。
    复核同时登记了纯记录问题 `V-38`（low）：两个历史导入测试错误地锁定活登记表的状态计数，
    在合法的向前状态变化后失败。本次 R-0024 导入把它们改为核对各自不可变的历史快照；
    不影响运行时、wheel、冻结的 H100 plan/job，A3-18 与 NEW-14 保持不变。
18. `R-0025` 已独立复核唯一一次 S-09 H100 作业 `10911504`。七项 CUDA-only
    测试通过；loop3d half/full 与 r1 half 三个 CUDA 长时程均完成 10,000 步到 `t=200`，
    因此 `A3-18` 在运行时 `85ba3ae…`、`V-38` 在纯记录提交 `07b0ea6…` 上推进为
    `verified`、`verified_by=R-0025`。累计 S-09 尚未完成：作业在 TF32-backward
    qualification helper 构造首个 runtime 前停止。
19. `R-0025` 新登记两个 qualification-support-only 问题：`V-39`（medium-low）要求
    TF32-backward helper 在调用 fixture 前显式创建各 case root；`V-40`（medium-low）要求把
    默认设备的三项 Consumer identity pin 核对与 CUDA finite/real-gradient smoke 分离。
    `NEW-14` 仍为 `open`，两项返工完成并经 CPU 全阶段演练后才可申请新的 H100 作业。
20. `V-39` 与 `V-40` 已由 qualification-support 提交
    `72286694c43ee15e5859c0658c989431404fe505` 修复并推进为 `claimed_fixed`。最终支持归档上的
    installed-wheel CPU 演练从环境身份、短时程、TF32、checkpoint、数值对照到独立 Consumer
    和 192 项 profile 规划全部通过；两项等待 `R-0026` 独立复核，`NEW-14` 仍保持 `open`。
21. `R-0026` 已独立重跑安装版 CPU 演练并验证 qualification-support 修复：`V-39` 的 12 个
    TF32 case 均在 fixture 构造前创建目录；`V-40` 的三项默认设备 identity pin、CUDA 信息性
    identity、可微 controls、实数损失和有限梯度边界均符合合同。两项在
    `72286694c43ee15e5859c0658c989431404fe505` 推进为 `verified`、
    `verified_by=R-0026`；`NEW-14` 仍为 `open`，S-09 须等待恢复 H100 作业验收后完成。
22. `R-0027` 已独立复核恢复作业 `10911538`：全部 H100 正确性阶段通过，`NEW-14` 在
    `85ba3aef75548d549d9db453ca4a5ab604b17a9d` 推进为 `verified`、
    `verified_by=R-0027`。S-09 仍未完成，因为 `channel_legacy` 与
    `channel_compiled_v2` 的 R512 单元 peak-reserved 比值为 `1.0307692307692307`，高于
    冻结上限 `1.03`。该问题登记为 `V-41`；用户已通过 `D-0012` 选择先做不作为资格的
    A·p 缓存开/关 H100 诊断，再决定是否撤回缓存并重跑完整作业，或另行裁决显存例外。
23. `R-0029` 撤回了 `R-0028` 基于单次 CPU `torch.profiler` 时间线作出的机制归因：
    完全相同的重复运行，其活跃峰值会变化一至两个压力场，因此 CPU 峰值只能作为参考，
    不能区分 CUDA 峰值路径上究竟是持久缓存还是局部 A·p 生命周期。增量大小仍为一个
    压力场，且由 `2573dec` 引入这两点不变；具体机制由两次不计时的 H100 allocator
    snapshot 裁决。精确撤回代理的可信度改由 `f94b9be` 方法源码逐字相同、候选命名空间
    编译、缓存字段为空、78 次算子调用与直接源码相同，以及终态逐位相同来证明。
24. `R-0030` 独立复核通过诊断作业的环境加固和冻结边界，同时发现纯记录问题 `V-42`
    （low）：四个历史证据测试错误地锁定活登记表在 V-41 之前的条数或状态集合，或者拒绝
    已由正式 successor 登记的源码哈希。本次导入改为只核对各历史记录自己的不可变快照，
    并显式解析向前 successor；不增加 skip/xfail，不改运行时、wheel、诊断 helper、冻结
    plan/job、案例或门限。`R-0031` 已将 V-42 验证为 `verified`，修复身份为 `f3620b8…`。
25. 用户通过 `D-0013` 选择有界例外并完成 S-09 裁决：例外只覆盖 `channel_legacy` 与
    `channel_compiled_v2` 的 `channel_R512x40x40`、float32、`peak_reserved_group_ratio`
    一项，接受作业 10911538 的 `1.0307692307692307`（562036736 / 545259520）；同一单元
    `peak_allocated=1.0150638559607594` 且全部计时门禁仍通过，其他单元和门限均不变。
    作业 10912583 与 `R-0031` 的 allocator snapshot 证明，一个 accepted A·p 持久缓存跨越
    `compute_nonlinear` 峰值存活。V-41 因而以设计取舍改为 `accepted`。同时登记记录勘误
    V-44：封存 summary 中不存在的 `616bc453a38b8f14…` 对应 Git 真实提交
    `616bc453d50040b20d1ef448cf2647358aeee1cc`；原归档不改写。
26. `R-0034` 发现本地 v1 结案重算错误绑定运行时树中的 plan `8bc4fb8d…`，而不是作业
    10911538 实际使用的 support plan `a21b211b…`，因此记录了 210 条 requirement failure 却
    错误声明非豁免门禁全部通过。该证据缺陷登记为 `V-45`（low）。v2 保留 v1 不动，改用
    正确 support 来源 `719ae760…`，并在失败消息不恰好等于两个 R512 float32 Channel
    `peak_reserved` 失败时立即失败关闭；`R-0035` 已逐字节复现 v2 重算并将 V-45 验证为
    `verified`，实际修复身份为 `e052f5b…`。
27. S-10 在独立分支上完成可独立验证的最后清理。`V-33` 的数值
    构建指纹升级为 schema 2：去掉仅随 CUDA 可见性出现或消失的四条 Torch 探测摘要，保留
    完整 build-settings 行，并改为哈希 NumPy 的结构化配置；v0.2.0 与 S-08 两份 oracle 的
    数值 payload 未改变。两套 D-0008 基线环境中，CUDA 可见/隐藏指纹均完全一致；v0.2.2
    环境在隐藏 CUDA 时会真实进入 S-08 的 matching-build 逐位门禁。`V-33` 推进为
    `claimed_fixed`，修复身份 `e75ef77…`，等待 `R-0032`。
28. `R1-R9` 按 `D-0011` 只完成两项可独立验证的子集：三套 public compiler 的 exact-key
    验证机制统一到标准库-only core helper（R7 partial）；`api`、`boundaries`、`diagnostics`
    与 `functional` 四层纳入 executable import-boundary ratchet，并纠正 charter 的债务数量
    说明（R9 partial）。其余部分保持 `deferred`：R1/R2/R3 会改变构造、checkpoint 或独立
    Consumer 公共边界；R4/R5 触及数值和已资格化 hot path；R6 需要弃用与 lifting 决策；
    R8 涉及证据保留政策；其余 R7/R9 需要更广泛迁移。登记决定改为 `D-0011`。
29. `R-0032` 对 S-10 的 R7、R9 部分清理和 D-0011 延期边界判定通过，但要求返工
    `V-33`：第一版只过滤 pip Torch 的 CUDA 探测行，v0.2.0 oracle 又混合了两个环境的
    指纹，因此不能证明真实 matching-build 分支。另登记 `V-43`（low）：NumPy 1.22.3
    不支持 `np.__config__.show(mode="dicts")`，而它仍是公开依赖下限和 HPCC 资格环境。
    `61838af…` 改为只绑定 Torch 的唯一 build-settings payload 和显式版本身份，并为旧
    NumPy `show()` 签名加入带来源字段的 fallback；两套真实生成环境都已显式进入 bitwise
    分支。`V-33`、`V-43` 现均为 `claimed_fixed`，等待 R-0033。
30. `R-0033` 已独立验证 S-10 返工：conda-forge 与 pip 两种 Torch 构建中的数值构建
    指纹均不随 CUDA 可见性变化；v0.2.0 oracle 和 S-08 fixture 使用各自真实生成环境的
    指纹，七个续算案例在匹配环境中全部进入逐位分支，payload 未改变；Python 3.10、
    NumPy 1.22.3、torch 2.5.1 环境中的 `show_text` fallback 与 S-09 checkpoint 阶段测试
    通过，而旧实现负向对照按预期失败。`R-0033` 复核的原始提交是 `61838af…`；最终
    S-09/S-10 线性集成中，patch-id 与 `pssolver` 树相同且可由分支到达的重放提交是
    `a540740…`，因此 `V-33`、`V-43` 的 `fixed_in` 前向更正为后者，验证身份仍为
    `verified_by=R-0033`。按 `D-0013`，S-10 无需另跑 H100；本次集成不改变 S-09 的
    资格结论。
31. `R-0036` 在只含分支可达对象的新克隆中发现 `V-46`（low）：第一版 S-09/S-10
    集成仍把原 S-10 的 10 个提交作为实时身份使用；这些对象只存在于本地 reflog，导致
    四个身份/集成测试失败。记录返工把 successor 58 的完整原提交→重放提交映射作为
    明确前向更正，测试只依赖可达提交和运行时树，并要求 `ff70b4c…` 到候选头恰好是九个
    固定重放提交加当前集成提交。原 S-10 头仅保留在本地
    `refs/archive/s10-pre-integration`，不推送。`V-46` 现为 `claimed_fixed`，等待 `R-0037`。

## Control 路径

`C1-05`、`C1-10`、`C1-11`、`C2-18`、`C1-08`、`NEW-14`、`A3-18`
以及 `V-17`、`V-22`、`V-23`、`V-16`、`V-13`、`V-14`、`V-25`、`V-21`、`OBS-4` 会触及 Control
依赖或暴露的路径。当前 Control 研究没有执行这些已登记的 checkpoint 限制路径，所有条目的
`exercised_by_current_control_research` 均为 false。

## 0.3 路线图

S-07 的 6 个 low/info 条目与候选回归 `V-30` 均已由 `R-0018` 独立验证为
`verified`，S-07 完成；S-08 的 `A2-25`、`N5`、`V-31`、`V-32`、`V-34`
均已由 `R-0021` 独立验证，S-08 完成；`V-33` 与 `V-43` 已由 `R-0033`
独立验证，S-10 已完成并正在最终 S-09 之上做集成复核。
`A2-21/S2-12` 已完成 `S-05` 独立复核。`A3-18` 已由 `R-0025` 的真实 H100 长时程
门禁验证；`V-39` 与 `V-40` 已由 `R-0026` 验证，`NEW-14` 已由 `R-0027` 验证。
S-09 已由 `R-0035` 完成结案复核：`V-41` 按 `D-0013` 接受窄范围例外，`V-42` 与
`V-45` 已验证，`V-44` 保留为封存证据勘误。
`V-46` 正在进行只涉及记录与测试的集成返工。
`V-27` 归入 `S-06`，`V-26` 与
`B30` 归入文档批次 `S-07`。
`V-17` 已验证身份不变，不重新打开。路线图顺序以同步仓库中用户确认的
`D-0010` 与 `plan/roadmap.json` 为准。

## 延期与覆盖缺口

- lifting：`B8/S2-07`，按已确认决定 `D-0007` 在重新开放 lifting 前处理；
- 架构整理：`R1-R9`，按 `D-0007` 推迟至 0.3 之后；
- `A2-11`：按 `D-0007` 接受为设计限制；
- CUDA 长时程覆盖缺口 `A3-18` 已由 `R-0025` 关闭；qualification-support 问题 `V-39`、
  `V-40` 已由 `R-0026` 关闭；`NEW-14` 已由 `R-0027` 完成 backward TF32 资格验证。
  `V-41` 已按 `D-0013` 改为 `accepted`；`V-42` 已由 `R-0031` 验证；`V-44` 作为
  记录勘误保留在 0.3；`V-45` 已由 `R-0035` 验证。

机器可读的逐项标题、严重度、来源、版本影响、Control 影响、状态、目标版本和说明均在
`notes/pssolver_v0_2_x_open_issues.json` 中。路线图 S-09 已完成，当前进行 S-10 集成复核。
GPU / H100 集中资格任务；任何 GPU、Slurm 或 H100 执行仍需用户单独授权。
