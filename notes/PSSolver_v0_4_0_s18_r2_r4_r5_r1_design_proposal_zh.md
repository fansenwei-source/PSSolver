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

## 第二批：R4 stepping、谱刷新与进度恢复原语

主要文件是 `execution/state.py`、`integrators/state_backed.py`、
`step_program.py`、scalar reference，五类 runtime adapter/step，以及 Plane、
Periodic、Channel 的恢复工作流。共享层只提取精确顺序的控制原语，不把算术
重排成新的向量化实现。

必须保持字段更新顺序、in-place/alias 语义、real-first transform 顺序和次数、
callback 时机、刷新判断以及所有 progress/generation 计数。除 8×20 外，重点
测试多种 refresh interval、刷新前/当步/后一步 restart 和不一致进度的失败
关闭。torch.compile 图、CUDA transform 及 H100 性能/显存进入 S-19。

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

## 第四批：R1 model-geometry application registry

这是范围最广的一批。需要统一 capability、declaration、lowering、binding、
public compiler dispatch、application adapter、runtime construction 和 workflow
exposure 八层，但不得改变任何公开键、顺序、错误、identity 或数值结果。

先建立不接管执行的 shadow registry，逐项与旧分派比较；键、顺序、lowering、
binding、compiler identity、capability output 和负例全部精确一致后，才切换唯一
的内部 dispatch 入口。近邻错误键、预分配失败路径、installed-wheel 构建和
全部 checkpoint reader 都是硬门禁。GPU-capable pair 的完整构建、图、梯度、
checkpoint/restart、性能和显存进入 S-19。

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
