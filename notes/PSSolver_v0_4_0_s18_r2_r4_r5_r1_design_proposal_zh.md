# PSSolver v0.4.0 S-18：R2、R4、R5、R1 分批设计草案

本记录只冻结设计，不实施运行时改动。建议按风险从低到高依次处理
`R2 → R4 → R5 → R1`，每批使用独立分支、独立复核请求和单独的推送
授权。`D-0018` 目前是 `proposed`；Claude 验收设计、用户确认该决定之前，
不得开始 S-18 实现。

## 共同门禁

- 对照同时包括正式 `v0.3.0` 和每批开始时已经验收的 `dev/0.4` 父提交。
- 默认且当前唯一登记的数值要求是逐位相同；当前没有登记任何非零容差。
- 如果最小原型证明逐位相同做不到，立即停止。在同一环境里先测
  `v0.3.0` 自比较、批次父提交自比较和重复试验的基线波动，再另写 proposed
  决定，明确比较对象、度量、全部样本、阈值和依据。用户确认之前不继续。
- 不放宽已有的数值、科学、restart、性能、显存或打包门限；不新增
  skip/xfail；不改变生产默认值。
- 每批都在只含可达对象的新克隆里跑 conda 与 D-0008 a2 两套完整 CPU
  suite，并在屏蔽仓库外证据路径时跑完整 conda suite。
- 每批都做 8 个标准 Plane/Periodic/Channel family × 20 步比较：每一步状态
  `torch.equal`，最终 runtime/configuration identity 精确相同；另做干净归档
  wheel 的导入、`pip check` 和标准 dry run。
- 每一批都必须让 PSSolver-Control main 对候选 installed wheel 跑完整 CPU suite；
  即使该批把 Control-facing 文件归为“只读冻结”，也不能省略这道门禁。
- 全部已发布的 v0.2.x 与 v0.3.0 checkpoint 都保持只读兼容，源目录不得被
  修改。

## 第一批：R2 checkpoint 与 integrity helper 合并

主要文件是 `pssolver/io/checkpoint.py`、`checkpoint_identity.py`，functional 的
Periodic/Channel checkpoint 与 identity，以及 workflows 的 Plane、Periodic、
legacy Channel、complete-stress Channel checkpoint；finite-Q 与 Robin scalar 的
已发布载体也纳入兼容边界。

共享实现只放在 `pssolver.io` 的私有层。格式号、metadata 字节、payload、异常
类型与消息、原子发布语义、迁移语义和 restart 结果全部冻结。验收覆盖仓库内
全部历史 fixture、由 v0.3.0 tag 生成的六类现行格式、当前 writer 的目录树逐
字节比较、故障/篡改矩阵和 N-1 合同；只有确实触及发布实现时才在单独授权下
复测真实 NFS。CUDA device/cross-device checkpoint/restart 进入 S-19。

R2 的逐文件分类如下（同一行中的每个路径均单独受该分类约束）：

- **本批修改**：`pssolver/experimental/checkpointing.py`、
  `pssolver/experimental/modal_block_checkpoint.py`、
  `pssolver/functional/channel_checkpoint.py`、
  `pssolver/functional/checkpoint_identity.py`、
  `pssolver/functional/periodic_checkpoint.py`、
  `pssolver/integrators/reference_checkpoint.py`、`pssolver/io/checkpoint.py`、
  `pssolver/io/checkpoint_identity.py`、`pssolver/workflows/channel_checkpoint.py`、
  `pssolver/workflows/finite_q_anchoring.py`、
  `pssolver/workflows/periodic_checkpoint.py`、`pssolver/workflows/plane_checkpoint.py`。
  它们实际拥有当前 writer、digest/canonical JSON/tensor helper 或原子发布调用点。
- **只读冻结**：`pssolver/experimental/shadow_run.py`、`pssolver/functional/api.py`、
  `pssolver/functional/channel_activity_runtime.py`、`pssolver/functional/contracts.py`、
  `pssolver/functional/errors.py`、`pssolver/functional/periodic_activity.py`、
  `pssolver/runtime/finite_q_anchoring.py`、`pssolver/runtime/robin_scalar.py`、
  `pssolver/workflows/channel_active_nematics.py`、
  `pssolver/workflows/channel_beris_edwards.py`、
  `pssolver/workflows/periodic_beris_edwards.py`、
  `pssolver/workflows/plane_beris_edwards.py`、
  `pssolver/workflows/plane_compiled_v2.py`。这些是公开协议、内存载体、调用 facade、
  Control-facing bridge 或已发布 reader/restore plan；其源码、异常与行为均冻结。
- **范围外**：`pssolver/control/dal.py`；它是遗留 control-layer persistence，不是
  R2 的 production/Functional checkpoint 格式。

“触及发布实现”严格指改变 target 创建、staging 位置、placeholder ownership、
文件/目录 fsync、rename/renameat2 选择、清理、symlink/device 判定或这些调用图。
一旦发生，必须另行授权复测真实 NFS 的 Periodic/Channel 发布、restart、并发、
已有目标与 symlink 矩阵。只替换为逐字节和失败行为相同的 canonical JSON、digest、
manifest 或 tensor helper，且发布调用图不变，不单独触发真实 NFS。

## 第二批：R4 stepping、谱刷新与进度恢复原语

主要文件是 `execution/state.py`、`integrators/state_backed.py`、
`step_program.py`、scalar reference，五类 runtime adapter/step，以及 Plane、
Periodic、Channel 的恢复工作流。共享层只提取精确顺序的控制原语，不把算术
重排成新的向量化实现。

必须保持字段更新顺序、in-place/alias 语义、real-first transform 顺序和次数、
callback 时机、刷新判断以及所有 progress/generation 计数。除 8×20 外，重点
测试多种 refresh interval、刷新前/当步/后一步 restart 和不一致进度的失败
关闭。torch.compile 图、CUDA transform 及 H100 性能/显存进入 S-19。

R4 的逐文件分类如下：

- **本批修改**：`pssolver/execution/state.py`、`pssolver/integrator.py`、
  `pssolver/integrators/scalar_reference.py`、`pssolver/integrators/state_backed.py`、
  `pssolver/integrators/step_program.py`、
  `pssolver/runtime/channel_active_nematics.py`、
  `pssolver/runtime/channel_beris_edwards.py`、
  `pssolver/runtime/periodic_beris_edwards.py`、
  `pssolver/runtime/plane_beris_edwards.py`、
  `pssolver/runtime/plane_compiled_v2_step.py`、`pssolver/runtime/plane_legacy.py`、
  `pssolver/workflows/channel_beris_edwards.py`、
  `pssolver/workflows/periodic_checkpoint.py`、`pssolver/workflows/plane_checkpoint.py`。
  这些文件拥有实际 step、progress、refresh clock、legacy 动态谱刷新或恢复集成。
- **只读冻结**：`pssolver/experimental/channel_compiled_v2.py`、
  `pssolver/experimental/checkpointing.py`、`pssolver/experimental/integrators.py`、
  `pssolver/experimental/modal_block_checkpoint.py`、
  `pssolver/experimental/modal_block_reference.py`、
  `pssolver/experimental/shadow_run.py`、
  `pssolver/functional/channel_activity_runtime.py`、
  `pssolver/functional/channel_checkpoint.py`、
  `pssolver/functional/periodic_activity.py`、
  `pssolver/functional/periodic_checkpoint.py`、
  `pssolver/integrators/reference_checkpoint.py`、
  `pssolver/runtime/plane_compiled_v2_binding.py`、`pssolver/runtime/robin_scalar.py`、
  `pssolver/workflows/channel_checkpoint.py`、
  `pssolver/workflows/plane_beris_edwards.py`、
  `pssolver/workflows/plane_compiled_v2.py`。其中 `periodic_activity.py` 与
  `channel_activity_runtime.py` 是 Control 使用的 Functional 路径，明确不修改，但
  R4 推送前仍须跑完整 Control main CPU suite。
- **范围外**：`pssolver/configuration/plane_beris_edwards.py`、
  `pssolver/experimental/_shadow_support.py`、
  `pssolver/experimental/h100_shadow_qualification.py`、
  `pssolver/experimental/plane_shadow_driver.py`、
  `pssolver/experimental/shadow_metadata.py`、
  `pssolver/planning/channel_compiled_v2.py`、`pssolver/planning/plane_compiled_v2.py`；
  它们分别属于配置、资格 support/driver/metadata、S-19 或 planning，而非执行原语。

## 第三批：R5 四条 Channel PCG 路径

四条路径是 production `_solve_pressure`、functional transpose、implicit
backward 和 unrolled oracle，位于：

- `pssolver/linear_solvers/stokes/channel_no_slip.py`
- `pssolver/functional/channel_pressure.py`
- `pssolver/functional/channel_pressure_adjoint.py`

私有共享层只统一停止、breakdown、roundoff saturation、rollback 和
postcondition 控制；operator/preconditioner 和 autograd 算术仍由四条路径的
callback 执行。调用顺序与次数、host sync、终止原因、迭代数、真实残差、异常
metadata、warm-start cache、值和梯度都必须精确相同。验收重放 S-06 的静默
发散与 floor-crossing 矩阵，覆盖 tolerance/fixed-work、float32/float64 及所有
失败路径。CUDA backward、profile、显存和性能统一进入 S-19，D-0013 的例外
不延伸到 v0.4。

R5 的三个**本批修改**文件就是上述三项；没有额外的只读冻结或范围外文件。

## 第四批：R1 model-geometry application registry

这是范围最广的一批。需要统一 capability、declaration、lowering、binding、
public compiler dispatch、application adapter、runtime construction 和 workflow
exposure 八层，但不得改变任何公开键、顺序、错误、identity 或数值结果。

先建立不接管执行的 shadow registry，逐项与旧分派比较；键、顺序、lowering、
binding、compiler identity、capability output 和负例全部精确一致后，才切换唯一
的内部 dispatch 入口。近邻错误键、预分配失败路径、installed-wheel 构建和
全部 checkpoint reader 都是硬门禁。GPU-capable pair 的完整构建、图、梯度、
checkpoint/restart、性能和显存进入 S-19。

R1 只修改通用 dispatch/registry 层：`pssolver/api/capabilities.py`、
`pssolver/configuration/simulation.py`、`simulation_binding.py`、
`simulation_lowering.py`、`public_simulation_runner.py`、
`pssolver/runtime/package_construction.py` 和 `simulation_construction.py`；允许新增私有
`pssolver/configuration/application_registry.py`，最终路径在实现前冻结。

以下组合专用文件全部归为**只读冻结**，registry 只能引用，不能改其源码：

- configuration：`channel_active_nematics.py`、
  `channel_active_nematics_declarations.py`、`channel_beris_edwards.py`、
  `periodic_beris_edwards.py`、`plane_beris_edwards.py`、
  `plane_beris_edwards_builders.py`、`plane_beris_edwards_component_graph.py`、
  `plane_beris_edwards_components.py`、`plane_beris_edwards_declarations.py`、
  `plane_beris_edwards_schema_v1.py`、`public_channel_beris_edwards_compiler.py`、
  `public_channel_simulation_compiler.py`、`public_periodic_simulation_compiler.py`；
- applications：`channel_active_nematics.py`、`channel_beris_edwards.py`、
  `periodic_beris_edwards.py`、`plane_beris_edwards.py`；
- planning：`channel_compiled_v2.py`、`plane_compiled_v2.py`；
- runtime：`channel_active_nematics.py`、`channel_application_bridge.py`、
  `channel_beris_edwards.py`、`periodic_beris_edwards.py`、
  `plane_application_bridge.py`、`plane_beris_edwards.py`、
  `plane_compiled_v2_binding.py`、`plane_compiled_v2_step.py`；
- workflows：`channel_active_nematics.py`、`channel_beris_edwards.py`、
  `periodic_beris_edwards.py`、`plane_beris_edwards.py`、`plane_compiled_v2.py`。

R1 的**范围外**文件是 `pssolver/configuration/finite_q_anchoring.py`、
`configuration/lifting.py`、`configuration/robin.py`、`pssolver/planning/lifting.py` 和
`planning/robin.py`：finite-Q/Robin 不是生产 registry 组合，lifting 继续按 D-0007
延期。所有路径的完整前缀与逐项理由以 JSON 的 `scope_classification_v2` 为准。

## Control-first 边界

S-18 不计划改变 `pssolver.functional.api` 的名称、语义或版本，也不计划改变
`beris_edwards.py` 和 `q_tensor.py` 两个 Control 固定的 git blob。任何一项若
变成必要改动，就立即暂停 Provider 批次，先在隔离 worktree 中迁移并 CPU
资格化 Control main；完成前 Provider 改动不得推送或发布。
`research/3d-warmup` 继续固定在 v0.2.0，不在本轮范围内。

## 等待用户确认的选择

`D-0018` 给出五个问题，均推荐 A：

1. `R2 → R4 → R5 → R1`；
2. 每批独立分支、独立验收、独立推送授权；
3. 现在不预授权任何非零容差，无法逐位时先停下再带实测数据决定；
4. R1 使用 shadow registry 后再切换；
5. 严格执行 D-0017 的 Control-first 边界。

本记录没有授权 push、tag、Release、PyPI、HPCC、Slurm、GPU 或 H100。
