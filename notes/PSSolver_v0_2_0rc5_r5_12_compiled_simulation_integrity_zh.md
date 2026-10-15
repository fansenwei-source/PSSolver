# PSSolver v0.2.0rc5 R5-12：CompiledSimulation 一致性封印

## 当前状态

实现已经完成，CPU 完整回归与 fresh installed-wheel 关闭验证仍在进行中。

## 修复

`compile_simulation()` 现在为编译产物生成一个稳定 SHA-256，绑定源 Simulation、应用身份、降级后的 application specification、lowering plan、construction plan、application request 的类型与内容，以及 normalization。`run_simulation()` 在选择 application runner 之前验证该身份；`to_metadata()` 在输出 provenance 之前也执行同一验证。

因此，用 `dataclasses.replace()` 替换 `application_request` 或其他任一已绑定字段时，旧封印不会随之改变，执行会以 `ValueError: compiled simulation integrity mismatch` 失败。拒绝发生在 runtime 构造、科学 timestep 和输出目录创建之前。

该设计不会在运行时重新编译，也不会在 timestep 热路径增加检查；每次 `run_simulation()` 只在进入 application runner 前验证一次。

## 范围

本阶段不修改科学方程、数值热路径、checkpoint schema、生产默认值、PSSolver-Control 或 nematics3d；不需要 H100，也没有 push。

