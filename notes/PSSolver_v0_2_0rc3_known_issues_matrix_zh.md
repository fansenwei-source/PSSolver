# PSSolver v0.2.0rc3 已知问题关闭矩阵

这份记录以已发布的 `v0.2.0rc3`（`5071d73a00e0d19be62ebd39edf9918818d1267a`）为不可变基线，把 Claude 的 rc1 审计和 rc2 修复复核转换为后续可执行的关闭队列。它不修改运行时代码，也不降低 rc3 已通过的 CPU、installed-wheel 或 H100 资格结论。

## 证据边界

两份输入审计的 SHA-256 已写入 JSON。`v0.2.0rc2..v0.2.0rc3` 之间真正影响数值运行时的文件只有：

- `pssolver/functional/channel_pressure.py`；
- `pssolver/functional/pressure_metadata.py`。

因此 N1 能依据 rc3 的新增回归测试和 H100 结果关闭；N2 由独立 PSSolver-Control 迁移和最终 installed-wheel H100 结果关闭。除此之外，rc2 复核中明确留下的缺口仍然适用于 rc3，除非后续用新的复现或实现提交将其关闭。

这并不表示 rc3 发布不合格。rc3 对它声明的 batch-one Periodic/Channel functional API、现有生产默认路径和四个已资格组合是合格的；它不等于整份历史审计清零。

## 当前统计

| 状态 | 数量 | 含义 |
|---|---:|---|
| `closed_verified` | 29 | 目标行为已有回归或资格证据 |
| `partial_residual` | 9 | 主问题部分缓解，但仍有路径或语义缺口 |
| `accepted_limitation` | 3 | 已知且明确接受，不作为当前正确性修复 |
| `deferred_architecture` | 9 | 架构债务，推迟到正确性关闭之后 |

由于仍有公开数值路径上的 `partial_residual`，当前不应把 rc3 直接提升为最终 `v0.2.0`。

## 修复顺序

### rc4.1：Plane Nyquist

只处理 B5。该批次现已完成：Plane free-slip Stokes 在偶数网格、`dealias_rule="none"` 下采用与 Periodic/Channel 一致的 Nyquist 约定；奇数网格保持逐字节不变，manufactured Stokes、两种谱存储等价性、CUDA correctness 与 R128/R320 H100 非回退门禁均已通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc412_h100_closure.json`。

把它单独作为第一个批次，是因为改动局部、物理判据清楚，且不会把 checkpoint 身份、lifting 数学和 PCG 合同混进同一次验证。

### rc4.2：checkpoint 与动力学身份

处理 B10、B22、N3、N5、N6。该批次已经完成：分层身份表达真实动力学版本，device token 与 fresh-only conditioning 不再错误阻断兼容续跑；legacy 输入按已登记 reader 接受或在 target mutation 前拒绝。installed-wheel CPU 门禁和 X01–X14 双向 CPU/CUDA checkpoint H100 矩阵均通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc426_h100_closure.json`。

### rc4.3：lifting 正确性

处理 B8、B20、N4。该批次已经完成：RC4.3.1 用真实 solver 与独立参考证明当前 DST remainder 不能支持公开的非零壁面 molecular-field 收敛声明；RC4.3.2–RC4.3.5 随后以 fail-closed restriction、显式 fresh-only conditioning、requested/effective metadata 和分层 checkpoint identity 关闭错误能力声明。RC4.3.6 installed-wheel CPU 门禁及 RC4.3.7 correct-layer CUDA、restart、R128/R320 性能与显存门禁均已通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc437_h100_closure.json`。这不表示原非收敛路径已被数学修复，也不重新开放 public lifted Plane。

### rc4.4：压力求解合同

处理 B25、B28、B29。该批次已经完成：fixed-work 与 tolerance 的优先级、active-mode RMS 极端尺度归一化、scale-relative breakdown、停止后置条件、diagnostics v2 和 scoped capability/runtime metadata 已在生产、functional 与 transpose 路径连接。PSSolver-Control 的梯度、Taylor、checkpoint、负向门禁以及单次 H100 累计资格均通过；R128×20×20 和 R512×40×40 的性能、显存与 transform-call 门禁通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc447_h100_closure.json`。

### rc4.5：跨 runtime 鲁棒性

处理 B7、B9、B11、B13、B15、B17、B21、B24、B27。RC4.5.0 已把它们拆成 checkpoint transactionality、boundary semantics、public validation/observability 和 accelerator global state 四个 cohort。RC4.5.1 随后完成十个 analysis-only probe：B7、B9、B11、B13、B15、B17、B24、B27 被黑盒重现并需要修复；B21 在 `1e14` 通过、从 `1e15` 失败，裁决为需要公开有限支持边界。RC4.5.1 未修改运行时、Control、nematics3d 或生产默认值，也没有提交 H100。权威规划与分析记录分别为 `PSSolver_v0_2_0rc4_rc450_cross_runtime_robustness_plan.json` 和 `PSSolver_v0_2_0rc4_rc451_cross_runtime_adjudication.json`。

### 0.2 之后的架构去重

R1–R9 不与上述正确性修复混做。优先顺序建议为 R1（组合注册）、R2（共享 checkpoint I/O）和 R5（统一 PCG），因为它们直接对应本轮出现的跨路径修复漂移。其后再处理 stepping、canonical helpers、qualification infrastructure 和 import boundary。

## Consumer 与外部项目边界

建立本矩阵不要求修改 PSSolver-Control 或 `nematics3d`。只有 Provider 改动触及 functional API、functional checkpoint identity、Channel pressure mathematics 或 Consumer 使用的公开声明时，才触发 Consumer-owned 迁移和重新资格验证。

## 下一项实现

`rc4.1_plane_nyquist`、`rc4.2_checkpoint_identity`、`rc4.3_lifting_correctness` 与 `rc4.4_pressure_contract` 均已完成；B25、B28、B29 已随 RC4.4.7 累计 H100 关闭转为 `closed_verified`。RC4.5.0 规划与 RC4.5.1 residual adjudication 也已完成。下一步是 RC4.5.2 checkpoint transactionality contract：先统一 B7、B24、B27 的 current-schema、public exception、atomic publication、caller-umask 与 cleanup 语义，再进入实现。当前分析结果不自行授权 RC4.5 源码实现、PSSolver-Control 修改、生产默认值、自动 merge、H100 或发布。
