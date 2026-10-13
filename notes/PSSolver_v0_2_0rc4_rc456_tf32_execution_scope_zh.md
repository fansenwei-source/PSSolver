# PSSolver v0.2.0rc4 RC4.5.6：TF32 执行作用域

## 结论

RC4.5.6 的本地 CPU 候选已经完成。PSSolver 不再把 TF32 设置作为构造或调用后的永久进程状态；八条已登记的生产与 functional runtime 只在自己拥有的执行区间内应用策略，并在所有退出路径上恢复调用者的精确状态。

分类：`PASS_RC4_5_6_SCOPED_TF32_CPU_CANDIDATE`

这不是 B11 的最终 CUDA 关闭。B11 在矩阵中继续保持 `partial_residual`，直至 RC4.5.7 installed-wheel CPU/Consumer impact 与 RC4.5.8 H100 门禁完成。

## 已实现的合同

- 进入作用域前先验证 TF32 请求、dtype 与设备，不合法请求不会改变进程状态。
- 快照并恢复 `torch.get_float32_matmul_precision()`、`torch.backends.cuda.matmul.allow_tf32` 和 `torch.backends.cudnn.allow_tf32`。
- 成功返回和异常退出均通过 `finally` 恢复。
- 可重入锁保证同线程嵌套作用域严格按 LIFO 恢复，并串行化 PSSolver 自身的并发作用域。
- dry-run/构造阶段只生成策略 metadata，不改变任何全局 flag。
- Plane legacy/compiled、Periodic complete-stress、Channel legacy/compiled/complete-stress，以及 Periodic/Channel functional runtime 均已连接。
- functional execution identity 明确记录有效 TF32 策略，因此后续 Consumer 必须按新 identity 做 installed-wheel 复核。

## 未改变的内容

- 科学模型方程、时间积分器、谱变换算法和压力算法未改变。
- checkpoint tensor 布局、生产默认值与公开 functional API 版本未改变。
- 未修改 PSSolver-Control 或 `nematics3d`。
- 本阶段未运行 H100，也未授权 push、自动合并或默认值提升。

## 本地验证

- focused TF32/runtime：226 passed，0 failed，0 skipped。
- 完整 CPU：3355 passed，1 deselected，8 subtests passed，0 failed，0 skipped。
- deselect 仅为既有 CUDA-only Plane Nyquist node ID；它必须在 RC4.5.8 的 H100 作业中真实运行。

权威机器可读记录：`notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json`。
