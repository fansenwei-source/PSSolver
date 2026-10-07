# PSSolver RC4.2.0 checkpoint 与动力学身份迁移方案

## 结论

RC4.2.0 的规划已经冻结，但尚未授权实现。它只处理 B10、B22、N3、N5、N6，不修改求解器热路径、checkpoint 文件、公开 API、生产默认值、PSSolver-Control 或 `nematics3d`，也不提交 H100 作业。

这批工作的核心不是“放宽 hash 检查”，而是把当前混在一起的身份拆开：

- run provenance 负责复现实验，保留请求 device、实际 device、初始化来源和输出策略；
- forward dynamics 只描述从当前状态到下一状态的物理与数值映射；
- derivative dynamics 额外描述 functional JVP/VJP、自定义伴随和隐式压力转置；
- state layout 描述持久张量的含义、顺序、shape、dtype 和谱表示；
- backend restart 描述必须持久化的算法状态和可重建缓存；
- materialization provenance 记录设备和分配位置，但不参与 restart compatibility。

只有会改变后续演化的内容才能否决 exact restart。device token、fresh seed、snapshot 路径和 `fresh_initial_remainder_conditioning` 不再属于兼容性身份；规定的 lift、边界值、线性 correction、Nyquist/Hermitian 约定、压力算法和积分器相位仍然属于兼容性身份。

## 为什么不能只删几个字段

现有 legacy checkpoint 通常只保存 `runtime_identity_sha256`，没有保存完整的 identity payload。SHA-256 不能反解，所以旧文件若只剩一个包含 device 或初始化字段的 digest，读取端不能猜它当时使用的是 `cpu`、`auto` 还是 `cuda:0`，也不能靠删除当前 metadata 中的字段证明旧 hash 的来源。

因此 RC4.2 必须同时提供两条路径：

1. 新格式直接保存分层、带 schema/version 的 identity，今后相同动力学可以跨 device 或 fresh initializer exact restart；
2. 旧格式由独立冻结的 legacy canonicalizer 验证。若去掉字段后无法仅凭旧文件唯一认证，必须要求经过认证的 source run spec，通过只写新目录的 upgrade tool 先复算旧 digest，再生成带迁移 provenance 的新 checkpoint。不得猜测，也不得原地改写旧 checkpoint。

B10 的修复同理：不能继续从今天的 functional identity 复制一份字典再 `pop()` 某个新字段来假装得到历史 schema。rc1/rc2/rc3 的 legacy identity canonicalizer 和 oracle 必须独立冻结，并由内容哈希约束。

## 五个问题怎样落到新模型

### B10：历史 functional identity

为 Periodic 和 Channel 建立独立的历史 identity registry。registry 以 checkpoint family、format version、source generation、runtime path 和 applicability class 为键。未知版本和未知 digest 必须在 tensor load 与 target mutation 前拒绝。

Periodic 的 `0.1-provisional` 不能继续被称为当前动力学的 exact restart。若保留兼容读取，它应是显式的 Hermitian reprojection state migration，并在结果中声明“不保持旧轨迹逐步等价”。

### B22：Plane device token

新 Plane checkpoint 的 forward dynamics identity 不含请求或实际 device。CPU→CUDA、CUDA→CPU、`auto`→显式设备只要其余动力学、state layout、backend restart 和 lifting identity 相同，就应允许 exact restart。

旧 v1 checkpoint 只有 opaque hash 时，不允许猜 source device；它只能在冻结 legacy digest 可复算时直接读取，或通过带 source run spec 的显式升级工具迁移。

### N3：Channel 动力学版本

rc2 修复了偶数周期网格的 forward Nyquist 行为，rc3 又修复了 functional pressure transpose 的 Nyquist 行为。新身份必须分别记录：

- production forward Stokes Nyquist semantics；
- functional pressure-transpose Nyquist semantics；
- implicit pressure adjoint 版本。

因此 rc1 的偶数周期网格 Channel checkpoint 不能静默作为 rc4 exact restart；rc3 以前的 even-grid functional checkpoint 也不能冒充当前 derivative dynamics。若网格为奇数且审计 oracle 已证明修复不适用，可以通过专门的 applicability class 接受，而不是粗暴按 package version 全拒绝。

### N5：lifted Plane 初值与真实动力学

`fresh_initial_remainder_conditioning` 只影响 fresh construction，恢复 checkpoint 时已有 homogeneous remainder，因此它必须从 restart identity 移出。但 rc1→rc2 的 lifting stress/B8 动力学变化必须进入 forward dynamics version；这样 rc1 lifted checkpoint 会因正确的动力学原因被拒绝，而不是被无关的初值调理字段偶然挡住。

prescribed lift、物理边界数据、lift representation 和 linear correction 仍然会改变未来演化，不能从 compatibility identity 删除。

### N6：Periodic Hermitian repair

identity 必须准确区分：

- full-complex：每一步投影完整共轭配对；
- Hermitian-half：每一步投影存储中的 self-conjugate planes。

rc1 pre-repair full-complex checkpoint 与当前 forward dynamics 不同，不能 exact restart。rc1 Hermitian-half 若已有逐字节 oracle 证明修复不适用，可以通过冻结 reader 接受。rc2/rc3 post-repair checkpoint 在其余身份一致时允许经冻结 reader 进入新格式。

## 跨版本矩阵

机器记录冻结了 M01–M16。关键判定如下：

| 来源与差异 | 预期 |
|---|---|
| 新 Plane 格式，仅 device 不同 | exact restart |
| 新 lifted Plane，仅 fresh initializer 不同 | exact restart |
| prescribed lift 或 linear correction 不同 | pre-mutation reject |
| rc1 lifted Plane 的 pre-B8 动力学 | 以 forward dynamics version 拒绝 |
| rc1–rc3 Plane 且 RC4.1 偶数 Nyquist 修复适用 | 拒绝 exact restart |
| rc1 Periodic full-complex pre-Hermitian repair | 只允许另行资格化的 state migration |
| rc1 Periodic Hermitian-half 且 oracle 证明不受影响 | 冻结 legacy reader 接受 |
| rc1 even-grid Channel pre-Nyquist repair | 拒绝 exact restart |
| rc1 odd-grid Channel 且修复确定不适用 | 冻结 legacy reader 接受 |
| pre-rc3 even-grid Channel functional derivative | 拒绝 functional exact restart，或显式 state-only migration |
| 未注册 schema/digest | pre-mutation fail-closed |
| 不同 runtime path | 保持现有拒绝，不在 RC4.2 偷渡扩展 |

“接受”不等于忽略版本，而是由明确的旧 schema 和 applicability oracle 证明两端具有相同下一步映射。“迁移”不等于 exact restart，必须单独记录状态修复和轨迹不连续边界。

## 恢复顺序与原子性

实现必须按以下顺序完成：metadata 完整性 → family/format → legacy/current schema → runtime family/path → forward dynamics → functional derivative dynamics → state layout → backend restart → lifting dynamics → tensor 文件与 checksum → shape/dtype/finite → 完整 copy/migration plan → 第一次 target mutation → progress/caches。

任何拒绝都必须保证：

- target tensor 未变化；
- pressure guess 等持久 solver state 未变化；
- completed steps 未变化；
- 不生成部分 migrated checkpoint 或输出目录。

## 分阶段实现

1. **RC4.2.0（本次）**：冻结分层身份、legacy 策略、M01–M16 和门禁。已完成。
2. **RC4.2.1**：只加入 typed identity primitives、frozen legacy registry 和 oracle fixtures；不改变 reader/writer 接受行为。
3. **RC4.2.2**：迁移 Plane checkpoint，解决 B22/N5，并提供只写新目录的 legacy upgrade path。
4. **RC4.2.3**：为 Periodic/Channel production forward dynamics 建版本，覆盖 RC4.1、Hermitian repair 和 Channel Nyquist。
5. **RC4.2.4**：冻结 functional identity registry 与 derivative dynamics；该步会触发 PSSolver-Control installed-wheel/checkpoint/gradient 重新资格。
6. **RC4.2.5**：installed-wheel CPU 跨版本矩阵、连续/恢复等价、pre-mutation negative gates 和完整 CPU suite。
7. **RC4.2.6**：只有在声称真实 CPU↔CUDA portability 时，执行一次 H100 checkpoint smoke；不需要性能 profile，也不据此改变生产默认值。

每个切片必须单独提交、单独关闭，不能把 Plane、Periodic、Channel、functional 和 Consumer 一次性混成一个不可归因的大提交。

## 当前授权边界

当前只确认 RC4.2.0 planning 完成。尚未授权 RC4.2.1 或后续代码实现、legacy 文件原地改写、PSSolver-Control 修改、H100 submission、cross-runtime restart 扩展、lifting 数学或 pressure contract 修改、自动 merge、tag、release 或默认值提升。
