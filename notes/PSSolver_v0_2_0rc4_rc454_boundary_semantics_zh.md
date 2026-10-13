# PSSolver v0.2.0rc4 RC4.5.4 boundary semantics

## 结论

`PASS_RC4_5_4_BOUNDARY_SEMANTICS_CPU_CANDIDATE`

RC4.5.4 已依照 RC4.5.2 冻结合约实现并关闭 B9 与 B21。该阶段只修改 finite-Q normal 的物理等价语义与 Robin 已公开支持范围的 fail-closed 边界，没有改变模型方程、时间推进、谱变换、压力算法或生产默认值。

## B9：finite-Q normal 的正反号等价

uniaxial Q 对 director 的符号为偶对称，因此与某个 face outward normal 平行或反平行的单位向量现在被视为同一物理声明。两者都归一到 oriented outward face normal，获得相同 metadata、canonical SHA-256、Robin policy 与逐字节相同的 target Q。

非平行、零长度和非有限输入仍然 fail closed。错误消息稳定包含 axis、side、expected outward normal、observed normal 和允许的 parallel/antiparallel equivalence。原有 outward 输入生成的 target Q 数值不变。

## B21：Robin 支持上界

经几何长度缩放后的 lower/upper dimensionless impedance 的资格区间固定为闭区间：

`0 <= impedance <= 1e14`

非有限、负值和超过上界的值被拒绝。内部 root helper 抛出 `ValueError`；公共 `Simulation` 编译路径在 operator、runtime 或 output materialization 前把相同问题转换为带结构化 context 的 `PublicSimulationCompilationError`。这并不把实验性 Robin 组合提升为已资格化公共 runtime。

支持区间内的 root 数量、有限性、严格递增和区间归属由冻结测试验证。四组代表 impedance 使用 80 位独立参考值，normalized characteristic residual 不超过 `1e-12`，绝对 root 误差不超过 `5e-12`。pure Neumann 保留其数学上的零模；其他合格 roots 为正。既有 `(lower, upper)=(1,1)` 的 8-root tuple 逐 float 值不变。

## 验证和范围

RC4.5.4 专属测试为 19 passed、0 failed。完整 CPU 门禁为 3295 passed、1 deselected、0 failed、8 subtests passed；唯一 deselect 是要求真实 H100 device/storage identity 的 CUDA 测试。没有 skip、xfail 或额外 deselect。

RC4.5.0–RC4.5.3 历史记录没有改写；本记录通过 successor source 哈希绑定当前实现。PSSolver-Control、`nematics3d`、B13/B15/B17、B11、生产默认值和 public Robin promotion 均未修改。没有运行 H100，没有 push、merge 或发布。

RC4.5.4 完成后具备进入 RC4.5.5 public validation/observability implementation 的资格，但本记录不授权 RC4.5.5、Control 修改、H100、自动 merge 或默认提升。
