# PSSolver v0.2.0rc5 R5-13 可选卫生加固

R5-13 处理合并审计中剩余的低风险卫生项，但不扩大数值、物理或 lifting 范围。

已实现：

- checkpoint 目录使用不覆盖发布，并在发布前后执行必要的 `fsync`；三个临时 writer 共用该事务；
- Plane restart 拒绝进入结构化公共错误边界，dry-run 不再绕过 restart 元数据和绝对保存窗口检查；
- 通用 CUDA 测试不再硬编码 H100，只有显式 `PSSOLVER_REQUIRE_H100=1` 的资格运行才要求 H100；
- PDF 通过 `.gitattributes` 明确视为 binary。

兼容性裁决：

- 根导出的 `DEFAULT_SPECTRAL_STORAGE` 属于冻结 v0.1 表面，保留；
- legacy bridge metadata version `1` 是 rc1–rc3 checkpoint identity 的组成部分，保留；当前 bridge format 常量仍为 `3`；
- lifting 仍然 fail-closed，因此不为 N4 重开 lifting；
- 历史 whitespace 不机械改写，以免破坏发布与证据哈希。

所有历史记录保持原文不变；变化仅通过 R5-13 的 forward-only successor registry 解析。

本地验证结果：定向门禁 252 passed；补充 successor 解析门禁 47 passed；完整 CPU suite
3615 passed、8 subtests passed、0 failed、0 skipped、0 xfailed。R5-13 不需要 H100，
没有 push，也没有修改 PSSolver-Control 或 nematics3d。

Fresh installed-wheel CPU closure 也已完成：49 项定向测试通过；适用完整 suite 为
3601 passed、7 个精确 CUDA-only skip、8 subtests passed、0 failed。源码布局专属的
7 项审计被明确排除而非伪装成 skip。`pip check` 通过，`pssolver` 从新 venv 的
`site-packages` 导入，没有 source shadow import。归档位于：

`/home/fansenwei/pssolver_rc5_r513_installed_cpu_0bcdd32_20261015_v1`
