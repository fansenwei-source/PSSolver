# PSSolver v0.2.0rc2 bug 审计修复记录

## 1. 范围与身份

本记录对应 `PSSolver_v0_2_0rc1_bug_redundancy_audit_zh.pdf` 中的直接缺陷和公共 API 问题。

- 冻结基线：`v0.2.0rc1`，提交 `1c5237194c5e4f696bb5d3d01d2caa2c4bdd27a1`
- 修复分支：`fix/v0.2.0rc2-audit`
- 独立工作树：`/home/fansenwei/Desktop/Develop/PSSolver_v0_2_0_rc2_audit`
- 原始脏工作树和 `PSSolver_release_v0_2_0_rc1` 均未修改
- 本地修复期间没有 push、merge、发布或修改生产默认 runtime

本批次处理审计建议中的第一批和第二批直接缺陷。第三批 R1–R9 结构清理以及旧审计的冗余清理是独立的架构迁移，不与 rc2 行为修复混合。

## 2. 直接缺陷处理结果

| ID | 状态 | 修复提交 | 结果摘要 |
| --- | --- | --- | --- |
| B1 | 已修复 | `3739a99` | 每步将 periodic `full_complex` 状态重新投影到 Hermitian 子空间，覆盖自共轭平面漂移。 |
| B2 | 已修复 | `5e1ef3e` | Channel dealias 后重新保持速度无散，补充投影与 full/truncated 一致性回归。 |
| B3 | 已修复 | `8d85fb2` | compiled Plane 按解析 dtype 构造和比较 activity，float32 可运行。 |
| B4 | 已修复 | `3330423` | 保存观测前同步静态流体场，Q/u/p 对应同一时间步。 |
| B5 | 已修复 | `faff73f` | 偶数网格、`dealias=none` 时清除不可解析的 Stokes Nyquist 导数模。 |
| B6 | 已修复 | `7338d29` | Periodic/Channel 编译期拒绝尚未实现的 `node_centered` runtime，不再静默忽略。 |
| B7(i)(ii) | 已修复 | `d2516dc` | checkpoint 导入验证文件记录、SHA、shape、dtype、有限性和 schema，再暴露恢复结果。 |
| B7(iii) | 已修复 | `cca4110` | Channel restore 在验证 progress 和所有输入后才修改目标状态。 |
| B8 | 已修复 | `1451f85` | lifted molecular field 使用与壁面余项兼容的基投影，消除错误 DST 近壁处理。 |
| B9 | 已修复 | `fb924c0` | anchoring helper 验证 face normal 与 geometry wall axis 一致。 |
| B10 | 已修复 | `85c8768` | Periodic 动力学身份修复进入新 checkpoint identity/version，旧 provisional 身份不再被误认。 |
| B11 | 已修复 | `f49e1f2` | Periodic、Channel 和 Plane 的生产 application 都显式执行同一 TF32 policy。 |
| B12 | 已修复 | `6f545ee` | public runtime flags、布尔值和 validation SHA 改为严格解析，字符串不再被 truthiness 接受。 |
| B13 | 已修复 | `d5e18b1` | device 和压力求解选项在创建输出/runtime 前验证，错误输入不再留下部分目录。 |
| B14 | 已修复 | `da29c5c` | restart 的保存窗口按绝对终止步 `start_step + steps` 校验。 |
| B15 | 已修复 | `c7e7780` | Periodic/Channel diagnostics 同时持久化文件和 metadata，不再只存在于返回对象。 |
| B16 | 已修复 | `cff45f8` | disabled spectral refresh 在四种公共组合中使用统一 canonical 表示。 |
| B17 | 已修复 | `d129550` | capability catalog、binding 和 compiler 的拒绝语义与异常类型一致。 |
| B18 | 已修复 | `28533b6` | Plane restart 不再生成或写入与实际恢复状态无关的缺陷气初态产物。 |
| B19 | 已修复 | `9741542` | lifted Plane metadata 从有效 boundary/lifting plan 派生，不再错误声明 Neumann/free。 |
| B20 | 已修复 | `1464852` | 初态余项在投影前满足 wall compatibility，抑制截断正弦展开的近壁 Gibbs 污染。 |
| B21 | 已修复 | `49e5bc2` | Robin 弱阻抗根在浮点 bracket 失效时采用受约束渐近定位。 |
| B22 | 已修复 | `57beaa8` | lifting checkpoint identity 不再绑定具体 device 字符串，支持合法跨设备恢复。 |
| B23 | 已修复 | `8d1cabb` | 非 CUDA functional device identity 规范化为实际分配设备类型。 |
| B24 | 已修复 | `049a2a9` | functional checkpoint 的格式、兼容性、完整性和 I/O 失败进入稳定异常分类。 |
| B25 | 已修复 | `5f63400` | Channel pressure adjoint 对极端 RHS 尺度归一化，同时保留正常尺度 byte oracle。 |
| B26 | 已修复 | `c811ae6` | functional identity 递归 thaw，支持 copy、deepcopy、pickle、replace 和字段重建。 |
| B27 | 已修复 | `2df1e08` | Periodic/Channel workflow checkpoint 原子发布前设为可读可遍历权限。 |
| B28 | 已修复 | `0c76951` | Channel functional declaration 与已验证 runtime、adjoint、checkpoint 能力一致。 |
| B29 | 已修复 | `f378f17` | 公开固定迭代、最大迭代和相对容差的机器可读优先级，不再静默覆盖。 |
| B30 | 已核实，不改代码 | — | `boundary_residual` 对构造基的近零结果是算子定义性质；审计也将其标为观察而非直接 bug，不能把它伪装成独立检测门禁。 |

相关 provenance 清单同步提交为 `afa63fb` 和 `df4ff01`。

## 3. 公共 API 遗留项

| ID | 状态 | 修复提交 | 决策 |
| --- | --- | --- | --- |
| A1 | 已修复 | `bf8d24f` | legacy Channel 编译期拒绝未实现的 snapshot 初态，避免编译成功后运行失败。 |
| A2 | 留待积分器架构阶段 | — | SBDF2 声明与 compiler 支持矩阵不一致，但不是 rc2 数值缺陷；应与积分器维度进入组合注册一起处理。 |
| A3 | 已修复 | `ee9857e` | public `SpectralNumerics` 和 `Output` 的分歧字段改为必须显式指定，入口不再暗含互相冲突的默认值。 |
| A4 | 已修复 | `135b62b` | legacy Channel 接受 geometry-neutral public `Output`，并显式适配其既有输出语义。 |
| A5 | 已核实，暂不改变物理参数路径 | — | 审计测得误差仅 1 ULP 且随机 round trip 全部通过；在建立 canonical parameter authority 前，不进行可能改变历史结果身份的局部重写。 |

A3 是发布候选阶段有意引入的 public API 收紧：旧代码若依赖这些隐式默认值，需要在调用端明确填写，但数值默认路径本身没有被偷偷改成另一种算法。

## 4. 审计中冗余问题的边界

以下内容没有在本修复分支中实施：旧审计 1–11 项和新增 R1–R9。它们涉及组合注册、checkpoint I/O 共用层、functional facade 收敛、统一半隐式更新、统一 PCG、实验模块归档、canonical/validation helper、资格产物治理和 import ratchet。其改动范围远大于兼容性修复，若与 rc2 bug 修复混合，会使数值回归和问题归因失去清晰边界。

建议在 rc2 行为修复完成资格验证后，按以下顺序另开架构分支：

1. R1：统一组合注册和 capability/compiler/runtime 路由；
2. R2：共享 checkpoint I/O；R5：统一 PCG；R4：统一半隐式更新；
3. 归档 Stage L–Q 并把 experimental import 改为惰性；
4. R8：收敛资格 helper 和 live-source hash 测试；
5. R6：隔离未进入生产的 Robin/finite-Q 试点；R9：修正 import boundary ratchet。

## 5. CPU 验证

固定环境：

- Python：`/home/fansenwei/anaconda3/envs/Nematics3D/bin/python`
- bytecode：`PYTHONDONTWRITEBYTECODE=1`
- pytest cache：禁用
- 登录节点精确 deselect 六个 CUDA-only node ID；这些测试不能被 CPU skip 冒充为资格证据

最终完整 CPU suite：

```text
2716 passed, 6 deselected, 8 subtests passed in 194.78s
```

结果中没有 failure、skip、xfail 或额外 deselect。`git diff --check` 通过，工作树在提交后保持 clean。

## 6. 尚需 GPU/H100 资格验证

本地 CPU 回归不能替代以下 GPU 证据：

1. B1 的 Periodic `full_complex` 长 horizon Hermitian 一致性与 finite gate；
2. B2 的 Channel dealias 无散性、full/truncated 等价性及性能/显存非回退；
3. B5 的偶数网格 Nyquist 处理在 CUDA 上的 manufactured Stokes gate；
4. B8/B20 的 lifting 数值、边界残差、显存和 restart；
5. B11 的实际 CUDA TF32 状态；
6. B25 的极端尺度 pressure adjoint CUDA finite/accuracy；
7. 六个登录节点 deselect 的 CUDA-only tests；
8. rc1 与本分支的 R128/R320 production smoke、checkpoint/restart 和 installed-wheel smoke。

在这些门禁完成前，本分支是“CPU 修复候选”，不应宣称已具备合并、默认值晋升或正式发布资格。
