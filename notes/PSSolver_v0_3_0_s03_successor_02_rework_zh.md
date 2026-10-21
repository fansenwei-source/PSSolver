# v0.3 S-03 异常兼容性返工候选

本层只返工 R-0008 重新打开的 C1-08、A2-24/C1-15，并在同一步处理新发现 V-24。C1-10、C1-11、V-23、V-21 保持 R-0008 的 `verified` 结论。

返工后的公共契约是：已有 `CheckpointError` 原样越过 production preflight/restore 边界；snapshot 读取错误仍是 `SimulationInputError`，同时保持 `OSError`，缺失文件还保持 `FileNotFoundError`；Periodic Functional checkpoint 导出与 Channel/Periodic 导入使用同一异常翻译函数。语义性 shape、dtype、finite 校验仍属于普通 `SimulationInputError`。

本候选不修改 checkpoint schema/bytes、provider identity、方程、数值算法、生产默认值或测试阈值，也不授权 push、tag、Release、HPCC、GPU 或 H100。
