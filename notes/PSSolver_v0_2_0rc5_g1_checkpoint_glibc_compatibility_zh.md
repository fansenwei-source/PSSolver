# PSSolver v0.2.0rc5 G-1.10：glibc 2.17 checkpoint 发布兼容修复

本修复解决 Job `10885517` 暴露的 checkpoint 文件系统兼容性缺口，不改变 checkpoint schema、身份、科学状态或生产默认值。

原实现只调用 glibc 的 `renameat2` 包装符号。HPCC 的 glibc 2.17 不导出该符号，但现场只读诊断确认其 Linux x86‑64 内核实际支持 syscall 316：使用 `RENAME_NOREPLACE` 的临时目录探针返回 0，并原子地把 source 发布为 target。

新实现仍优先使用 glibc wrapper；仅在 wrapper 缺失且机器为 `x86_64`/`amd64` 时调用同一个 `renameat2` 内核 syscall。它没有增加“先检查目标、再普通 rename”的竞态降级。目标抢占仍返回 `CheckpointExistsError`；内核返回 `ENOSYS` 或架构未知时继续 fail-closed。

新增测试覆盖旧 glibc 成功路径、空目录竞态拒绝、内核 `ENOSYS` 和未知架构。核心兼容/事务/历史记录集合为 `65 passed, 2 skipped`；整个非 CUDA checkpoint 集合为 `318 passed, 0 failed`；完整 CPU suite 为 `3716 passed, 1 skipped, 7 deselected, 8 subtests passed, 0 failed`。受影响的 RC4.5.x 与 R5-1 至 R5-13 历史记录保持逐字节不变；各静态门禁只通过新的 forward-only successor record 继续解析当前源码哈希。

本阶段只完成 CPU compatibility candidate。没有运行 H100、没有 push，也没有修改 PSSolver-Control 或 nematics3d。下一步是在 HPCC 登录节点用 fresh installed wheel 执行 glibc 2.17 checkpoint smoke；完成后才能请求新的 push 与单次 H100 recovery 授权。
