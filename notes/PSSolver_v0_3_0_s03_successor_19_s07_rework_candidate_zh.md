# PSSolver v0.3 S-07 返工候选

R-0017 发现的 V-30 已在 `8aceaab` 中修复：Channel 共用的 active-force 实现移入随 wheel 打包的 active-nematics 模型命名空间，旧 `pssolver.control.active_force` 只保留源码兼容转发。安装版 wheel 的全部模块导入、两种 Channel runtime 构造以及 Plane、Periodic、legacy Channel、complete-stress Channel 四条真实 public dry-run 均已通过。

同时，checkpoint 发布完成后的 OBS-4 日志发送改为 best effort；用户 logging filter 的异常不会再把已经完成的持久化误报为失败。此变化不进入 checkpoint identity，也不改变 checkpoint schema、provider identity、数值算法、生产默认值或已有阈值。

V-30 当前为 `claimed_fixed`，等待 R-0018 独立复核。其余六项 S-07 claim 保持不变。本地完整 CPU 门禁将在最终候选封存前执行；当前没有 push，也没有运行 HPCC、GPU 或 H100。
