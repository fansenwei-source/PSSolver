# R-0028 返工导入与精确撤回代理

R-0028 的分类是 `REWORK_REQUIRED`。独立复核确认运行时与 wheel 不需要修改，
但原 cache-off harness 不能代表真实撤回：R512 上它与 cache-on 的 CPU 活跃内存
峰值相同；真正 `f94b9be` 源码少一个压力场。

资格支持提交 `98a3e26eb3d3ac0567079b1e0ee1e5c22b998ad2` 与 analyzer
bootstrap 后继 `b0314216d1943c5131097f80d0ec164fe1b06bd2` 因此：

- 精确重建并校验 `f94b9be` 的 `_solve_pressure`，在候选模块命名空间临时编译；
- 将 72 项矩阵第三角色改为 revert-proxy；
- 六次 trial 使用全部 6 种排列；
- 单 trial 比值改用冻结 `_profile_center`；
- 增加两次独立、不计时的 R512 CUDA allocator snapshot；
- 在 V-41 中登记局部 A·p 生命周期归因。

运行时仍是 `85ba3ae`，两个 wheel 不变；V-41 保持 `registered`，S-09 仍未完成。
本记录不授权 push、HPCC、Slurm 或 H100。
