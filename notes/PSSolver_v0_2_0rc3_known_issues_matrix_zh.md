# PSSolver v0.2.0rc3 已知问题关闭矩阵（含 v0.2.0rc4 发布后 successor）

这份记录以已发布的 `v0.2.0rc3`（`5071d73a00e0d19be62ebd39edf9918818d1267a`）为不可变基线，把 Claude 的 rc1 审计和 rc2 修复复核转换为后续可执行的关闭队列。2026-10-14 的 successor 根据 v0.2.0rc4 发布后真实旧版 fixture 复核更新当前状态；它不改写任何历史 closure，也不修改运行时代码、tag 或 Release。

## 证据边界

原始两份输入审计及 v0.2.0rc4 发布后验证、Codex 草稿、Claude 盲审和合并报告的 SHA-256 均已写入 JSON。`v0.2.0rc2..v0.2.0rc3` 之间真正影响数值运行时的文件只有：

- `pssolver/functional/channel_pressure.py`；
- `pssolver/functional/pressure_metadata.py`。

因此 N1 能依据 rc3 的新增回归测试和 H100 结果关闭；N2 由独立 PSSolver-Control 迁移和最终 installed-wheel H100 结果关闭。除此之外，rc2 复核中明确留下的缺口仍然适用于 rc3，除非后续用新的复现或实现提交将其关闭。

这并不表示 rc3 发布不合格。rc3 对它声明的 batch-one Periodic/Channel functional API、现有生产默认路径和四个已资格组合是合格的；它不等于整份历史审计清零。v0.2.0rc4 的 fresh-run 资格证据也继续有效，但真实 rc1–rc3 functional checkpoint 的发布后复核推翻了 B10 的关闭状态，因此 rc4 的合并发布后分类为 `FAIL`。

## 当前统计

| 状态 | 数量 | 含义 |
|---|---:|---|
| `closed_verified` | 28 | 目标行为已有回归或资格证据，且发布后证据未推翻 |
| `partial_residual` | 9 | B7、B11、B13、B17、B20、B23、B24、N3、N6 仍有可复现残余 |
| `regressed_after_closure` | 1 | B10 曾关闭，但 rc4 拒绝真实 rc1–rc3 functional checkpoint |
| `accepted_limitation` | 3 | 已知且明确接受，不作为当前正确性修复 |
| `deferred_architecture` | 9 | 架构债务，推迟到正确性关闭之后 |

另登记 NEW-1…NEW-15 共 15 项发布后发现。RC4 历史批次均曾完成，但发布后状态不再允许把正确性队列视为全部关闭。rc5 的 W1 文档批次已经关闭 NEW-4、NEW-5，并完成 B30 的公开解释；其余修复继续按优先级执行。

## 发布后决定性回归

B10 的真实旧版 checkpoint fixture 位于：

`/home/fansenwei/pssolver_rc4_post_release_verification_20261013_v1/agent_evidence/state/legacy_ckpts_full/`

已发布 rc4 wheel 对 rc1、rc2、rc3 的 Periodic/Channel functional checkpoint 共六种组合，无论直接 import 还是显式 migration 都拒绝。`1b15bf8` 对 rc3 fixture 的对照仍能读入并逐位恢复，因此这是 `dc8c87a` / `a195bd0` 之后的 released regression，而不是 fixture 损坏。拒绝是 fail-closed，不会静默读错数据。

## 修复顺序

### rc4.1：Plane Nyquist

只处理 B5。该批次现已完成：Plane free-slip Stokes 在偶数网格、`dealias_rule="none"` 下采用与 Periodic/Channel 一致的 Nyquist 约定；奇数网格保持逐字节不变，manufactured Stokes、两种谱存储等价性、CUDA correctness 与 R128/R320 H100 非回退门禁均已通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc412_h100_closure.json`。

把它单独作为第一个批次，是因为改动局部、物理判据清楚，且不会把 checkpoint 身份、lifting 数学和 PCG 合同混进同一次验证。

### rc4.2：checkpoint 与动力学身份

处理 B10、B22、N3、N5、N6。该批次的历史 installed-wheel CPU 门禁和 X01–X14 双向 CPU/CUDA checkpoint H100 矩阵均曾通过，权威历史记录为 `PSSolver_v0_2_0rc4_rc426_h100_closure.json`。发布后真实 fixture 揭示：B10 已回归；N3 的 legacy Channel identity 仍可静默跨越偶数网格动力学变化；N6 的 full-complex 描述性元数据仍不准确。历史记录保持不变，当前状态分别为一项 `regressed_after_closure` 和两项 `partial_residual`。

### rc4.3：lifting 正确性

处理 B8、B20、N4。RC4.3.1 用真实 solver 与独立参考证明当前 DST remainder 不能支持公开的非零壁面 molecular-field 收敛声明；后续以 fail-closed restriction 关闭错误公开能力，原非收敛路径没有被数学修复，也不重新开放 public lifted Plane。B8 保持关闭；B20 因 strict fresh-conditioning 拒绝合同未实现而转为 `partial_residual`。权威历史关闭记录仍为 `PSSolver_v0_2_0rc4_rc437_h100_closure.json`。

### rc4.4：压力求解合同

处理 B25、B28、B29。该批次已经完成：fixed-work 与 tolerance 的优先级、active-mode RMS 极端尺度归一化、scale-relative breakdown、停止后置条件、diagnostics v2 和 scoped capability/runtime metadata 已在生产、functional 与 transpose 路径连接。PSSolver-Control 的梯度、Taylor、checkpoint、负向门禁以及单次 H100 累计资格均通过；R128×20×20 和 R512×40×40 的性能、显存与 transform-call 门禁通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc447_h100_closure.json`。

### rc4.5：跨 runtime 鲁棒性

处理 B7、B9、B11、B13、B15、B17、B21、B24、B27。历史 CPU、installed-wheel、Consumer 和 H100 门禁保持有效；但发布后更宽输入矩阵发现 B7、B11、B13、B17、B24 仍有残余，分别涉及 checkpoint 精确键与 reader 边界、`'medium'` TF32 恢复顺序、历史/functional 设备语法、结构化公开异常和错误分类。B9、B15、B21、B27 保持关闭。权威历史记录不改写，当前 residual 进入 rc5。

### 0.2 之后的架构去重

R1–R9 不与上述正确性修复混做。优先顺序建议为 R1（组合注册）、R2（共享 checkpoint I/O）和 R5（统一 PCG），因为它们直接对应本轮出现的跨路径修复漂移。其后再处理 stepping、canonical helpers、qualification infrastructure 和 import boundary。

## Consumer 与外部项目边界

建立本矩阵不要求修改 PSSolver-Control 或 `nematics3d`。只有 Provider 改动触及 functional API、functional checkpoint identity、Channel pressure mathematics 或 Consumer 使用的公开声明时，才触发 Consumer-owned 迁移和重新资格验证。

## rc5 执行队列

执行顺序固定为：

1. `rc5_documentation`：DOC1、DOC2、NEW-4、NEW-5、B30；已由 `PSSolver_v0_2_0rc5_w1_documentation.json` 完成并关闭；
2. `rc5_must`：B10、NEW-1、N3、NEW-2、NEW-3、B11；这是当前下一批；
3. `rc5_should`：其余九项 partial residual 与 NEW-7…NEW-14；
4. `rc5_optional` 与 release-process 卫生；
5. rc5 冻结后才执行累计 CPU、installed-wheel、真实旧版 checkpoint、Consumer 和经单独授权的唯一一次 H100 资格。

不得改动 v0.2.0rc4 tag/Release，不修改 `nematics3d`，不放宽阈值或增加 skip/xfail，也不得借文档修正重新开放 lifting。
