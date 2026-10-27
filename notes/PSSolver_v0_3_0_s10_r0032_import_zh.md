# PSSolver v0.3 S-10：R-0032 复核导入

R-0032 的分类为 `REWORK_REQUIRED`。R7 的 exact-key 机械集中、R9 的 import-layer
ratchet 扩展、D-0011 延期边界以及 S-09 分支隔离均通过。

V-33 保持 `claimed_fixed`。需要让 Torch 指纹不依赖 CUDA 可见性或构建输出的探测格式，
在真正的 Nematics3D conda-forge Torch 环境中重算 v0.2.0 oracle 指纹，并显式证明
v0.2.0 与 S-08 两条 matching-build 续算都进入 bitwise 分支。

新登记 V-43（low）：NumPy 1.22.3 不支持 `np.__config__.show(mode="dicts")`，但仍在
PSSolver 的公开依赖范围内。R-0033 前需要加入旧签名 fallback 及真实或模拟门禁。

本导入只改 notes、登记表和记录测试；不改运行时、oracle、S-09、HPCC 或排队作业。
