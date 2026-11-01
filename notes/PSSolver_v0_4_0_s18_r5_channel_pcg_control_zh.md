# PSSolver v0.4 S-18 R5：Channel PCG 控制流归并候选

本候选依据 D-0018，在不合并四条路径各自张量运算的前提下，把
Channel 压力 PCG 的停止、breakdown、fixed-work floor 回退与返回状态
postcondition 决策归入私有模块
`pssolver.linear_solvers.stokes._channel_pcg_control`。

涉及的四条路径为生产 primal、Functional transpose、implicit adjoint
primal/transpose 和固定迭代 qualification oracle。共享模块不执行张量运算，
不改变算子与预条件器调用顺序、host 同步点、异常、diagnostics、梯度或
warm-start cache 语义，也不导出为公共 API。

资格 helper 在独立子进程中记录返回张量、梯度、公开 diagnostics、warm
state、算子调用次数和有序 Torch dispatch 摘要。初步父提交对照已通过；
正式 clean-clone、完整 CPU、8×20、checkpoint、wheel 和 Control main
门禁仍须完成后才可提交独立复核。

本记录不授权 push、tag、Release、PyPI、HPCC、Slurm、GPU 或 H100。
