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
| `closed_verified` | 18 | 目标行为已有回归或资格证据 |
| `partial_residual` | 20 | 主问题部分缓解，但仍有路径或语义缺口 |
| `accepted_limitation` | 3 | 已知且明确接受，不作为当前正确性修复 |
| `deferred_architecture` | 9 | 架构债务，推迟到正确性关闭之后 |

由于仍有公开数值路径上的 `partial_residual`，当前不应把 rc3 直接提升为最终 `v0.2.0`。

## 修复顺序

### rc4.1：Plane Nyquist

只处理 B5。该批次现已完成：Plane free-slip Stokes 在偶数网格、`dealias_rule="none"` 下采用与 Periodic/Channel 一致的 Nyquist 约定；奇数网格保持逐字节不变，manufactured Stokes、两种谱存储等价性、CUDA correctness 与 R128/R320 H100 非回退门禁均已通过。权威关闭记录为 `PSSolver_v0_2_0rc4_rc412_h100_closure.json`。

把它单独作为第一个批次，是因为改动局部、物理判据清楚，且不会把 checkpoint 身份、lifting 数学和 PCG 合同混进同一次验证。

### rc4.2：checkpoint 与动力学身份

处理 B10、B22、N3、N5、N6。目标是让身份表达真实动力学版本，同时去掉 device token、fresh-initial conditioning 等不应阻断物理兼容续跑的因素。必须覆盖 rc1/rc2/rc3 的接受或明确拒绝矩阵，并保证在 target mutation 之前拒绝不兼容输入。

### rc4.3：lifting 正确性

处理 B8、B20、N4。这里不能只改 metadata：需要重新评估壁面 molecular field 的表示空间、非零壁面 Laplacian 以及初值 taper。如果当前 DCT/DST 结构无法给出所需收敛，应缩小公开能力或明确限制，而不是继续用不调用真实 solver 的投影测试代替端到端收敛。

### rc4.4：压力求解合同

处理 B25、B28、B29。统一 fixed-work 与 tolerance 的优先级、极端尺度归一化、停止后置条件和 public diagnostics。该批次会触及 functional Consumer 的数学合同，需要 PSSolver-Control 重新跑梯度、Taylor、checkpoint 和 H100 门禁。

### rc4.5：跨 runtime 鲁棒性

处理 B7、B9、B11、B13、B15、B17、B21、B24、B27。包括 checkpoint 降级/路径、TF32 全局状态、device 校验、异常类型、diagnostics schema、目录权限与原子发布。

### 0.2 之后的架构去重

R1–R9 不与上述正确性修复混做。优先顺序建议为 R1（组合注册）、R2（共享 checkpoint I/O）和 R5（统一 PCG），因为它们直接对应本轮出现的跨路径修复漂移。其后再处理 stepping、canonical helpers、qualification infrastructure 和 import boundary。

## Consumer 与外部项目边界

建立本矩阵不要求修改 PSSolver-Control 或 `nematics3d`。只有 Provider 改动触及 functional API、functional checkpoint identity、Channel pressure mathematics 或 Consumer 使用的公开声明时，才触发 Consumer-owned 迁移和重新资格验证。

## 下一项实现

`rc4.1_plane_nyquist` 已完成，B5 已转为 `closed_verified`。RC4.2.0 规划合同现已写入 `PSSolver_v0_2_0rc4_rc420_checkpoint_identity_plan.json`，冻结了分层身份、rc1/rc2/rc3 接受或拒绝矩阵、pre-mutation 拒绝顺序和分阶段迁移路径。当前仍未授权 RC4.2.1 实现，也不授权并行修改 lifting 或 pressure contract、改变生产默认值、自动 merge 或发布。
