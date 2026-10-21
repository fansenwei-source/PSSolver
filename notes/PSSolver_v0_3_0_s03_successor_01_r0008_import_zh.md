# S-03：R-0008 独立复核导入

R-0008 对本地候选 `b16b609` 的结论是 `REWORK_REQUIRED`。本登记不改写原 S-03 候选记录：C1-10、C1-11、V-23、V-21 已独立验证；C1-08 与 A2-24/C1-15 因异常类型兼容性回归重新打开；新增 V-24，记录 Periodic Functional checkpoint 导出未走统一异常翻译层。

后续返工只处理重新打开的两项与 V-24。已验证的四项保持 `verified_by = R-0008`，除非返工再次改动其行为，否则无需重复扩大复核范围。

本导入是 evidence-only：不授权 push、tag、Release、PyPI、HPCC、GPU 或 H100，也不修改 PSSolver-Control 或 nematics3d。
