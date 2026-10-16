# PSSolver rc5 G-1 timing instrumentation v12

状态：`READY_FOR_ANALYSIS_ONLY_TIMING_INSTRUMENTATION`

本记录冻结 v11 之后的下一步诊断工具。它不会重写 v10 的正式结论：

`FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE`

## 诊断设计

矩阵只运行候选 Provider，包含四个发生过计时不稳定的 runtime family，每个 family 三个全新子进程，共 12 个 profile。每个 profile 创建两个独立但初态相同的 runtime：

1. `asynchronous_compatibility` 保留连续 CUDA event 的 enqueue/同步拓扑，记录 60 个原始样本、host submit 时长、Python GC 事件和理论 refresh 相位。
2. `synchronized_attribution_replay` 每步同步，记录 end-to-end host 时长和可用的 public pressure diagnostics。该窗口只用于归因，不用于正式性能裁决。

两个窗口都记录绝对 host 时间戳。独立 `nvidia-smi` 子进程以 100 ms 周期采集 GPU UUID、物理 index、PCI bus、P-state、时钟、功率、温度和利用率；若 `CUDA_VISIBLE_DEVICES` 无法唯一映射到采样身份，输出必须标为 unresolved，不允许猜测。

每个 child 显式设置 `highest / TF32 false / TF32 false`，并在原子写入结果前恢复原有 Python runtime flags。每个 profile 使用独立空 Triton 和 TorchInductor cache。

## 不变边界

- 正式 raw sample 数仍为 60；
- 正式 CV 阈值仍为 5%；
- 不删除 outlier；
- 不替换 trial；
- 不修改 production solver、runtime 或默认值；
- instrumentation 会扰动时序，因此结果不能直接升级为 qualification PASS；
- GC、refresh、pressure 或 telemetry 的相关性都不能单独证明因果关系。

## CPU/synthetic 验证

在 HPCC 登录节点的 Python 3.10.18、PyTorch 2.5.1+cu121、pytest 9.1.1 环境执行：

- 新测试：9 passed；
- 新测试及相关既有门禁：45 passed；
- 0 failed，0 skipped。

验证覆盖 12 项矩阵、绝对 refresh 相位、GC callback 生命周期、telemetry 解析与设备绑定、installed-wheel command plan、完整 synthetic analyzer matrix，以及原子 JSON no-overwrite。

installed-wheel command preflight 接受解析到普通文件的标准 venv `bin/python` symlink；具体环境身份仍由既有 `lexical_python_path` 合同验证。preflight 还会 fail-closed 验证 Job 在输出目录创建之前依赖的 checkpoint `/tmp` helper、environment helper 与 profile helper，缺少任一文件都不得进入 `sbatch`。

`run_rc5_g1_checkpoint_tmp_preflight.py` 以此前 G-1 checkpoint recovery 已执行版本逐字节纳入仓库，SHA-256 为 `ec6b4d2c70f5b4454754120b0f87a8771878ec50787a5590e23759bcc23cf142`。它继续验证 installed-wheel import、独立 node-local `/tmp` 目录、原子 `RENAME_NOREPLACE` publication、并发目标拒绝和 staging residue。

## 授权状态

本轮没有提交 H100 Job，也没有 push。后续若要实际采集 12 项 instrumentation evidence，需要用户另行明确授权一次 analysis-only H100 submission；不得自动重试。
