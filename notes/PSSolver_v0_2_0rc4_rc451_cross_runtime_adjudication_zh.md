# PSSolver v0.2.0rc4 RC4.5.1 跨 runtime residual 裁决

## 结论

RC4.5.1 analysis-only 阶段已完成。十个冻结 probe 均已执行，未修改求解器运行时代码、数值算法、生产默认值、PSSolver-Control 或 nematics3d，也没有提交 H100 Job。

机器记录：`notes/PSSolver_v0_2_0rc4_rc451_cross_runtime_adjudication.json`

最终分类：

`RC4_5_1_RESIDUAL_ADJUDICATION_COMPLETE`

## 九项裁决

| Issue | 裁决 | 核心证据 |
|---|---|---|
| B7 | `REPRODUCED_REPAIR_REQUIRED` | Channel 与 Periodic functional checkpoint 都接受重新封印后的未知顶层键；当前 provenance 缺失会被拒绝，但 canonical SHA-256 明确只是完整性检查，不是认证机制。 |
| B9 | `REPRODUCED_REPAIR_REQUIRED` | `n` 与 `-n` 构造出的 Q 逐字节相同，但 lower face 只接受 outward `-z`，拒绝物理等价的 `+z`；错误信息没有 expected/observed vector。 |
| B11 | `REPRODUCED_REPAIR_REQUIRED` | `configure_tf32_execution()` 成功返回后没有恢复调用者的 matmul precision、CUDA matmul TF32 和 cuDNN TF32 状态。分析 harness 已在退出前恢复原状态；完整 CUDA/nested matrix 留给后续单次 H100 qualification。 |
| B13 | `REPRODUCED_REPAIR_REQUIRED` | Plane 的 `legacy_production` 与 `compiled_v2` 在 public compile 阶段都接受 `device="banana"` 和 `device="cpu:0"`。 |
| B15 | `REPRODUCED_REPAIR_REQUIRED` | 四个 public application family 的 diagnostics 字段集合不一致；Plane/legacy Channel 使用 `div_*`/`schur_*`，complete-stress family 使用 `divergence_*`/`pressure_*`，迭代次数还存在 float64/int64 差异。 |
| B17 | `REPRODUCED_REPAIR_REQUIRED` | Periodic 的非法 device 已转换为 `PublicSimulationCompilationError`，但 Plane 的非法 pointwise/reuse 仍分别泄漏原始 `ValueError`/`TypeError`。Periodic no-slip 会被显式归一化为 qualified periodic equivalent。 |
| B21 | `PUBLIC_BOUND_REQUIRED` | symmetric impedance 在 `1e14` 仍得到八个 roots，从 `1e15` 起稳定无法 bracket。此时 finite-Robin correction 已接近 binary64 对 Dirichlet-limit roots 的分辨率；建议公开并强制 `0 <= impedance <= 1e14`。 |
| B24 | `REPRODUCED_REPAIR_REQUIRED` | missing、existing、future-format、payload tamper 已进入 functional public exception hierarchy；但父路径是普通文件时仍泄漏 built-in `FileExistsError`。 |
| B27 | `REPRODUCED_REPAIR_REQUIRED` | 相同 `umask 077` 下 functional Channel 发布目录为 `0700`，functional Periodic 为 `0755`；注入 writer failure 后 staging cleanup 正常，三个 production writer 均使用 staging + `os.replace`，但 mode 策略仍不统一。 |

## 已确认但不能关闭整个 issue 的正向证据

- C01：current functional provenance 缺失会在 tensor payload 打开前拒绝。
- C02：Plane、legacy Channel、Periodic production tensor reader 均拒绝 `../escape.npy`。
- C03：writer failure 不留下 target 或 staging residue。
- C04：existing target、missing target、future format 与 payload tamper 的主要异常类型已经稳定。
- V02：Periodic invalid-device 已使用 public compilation error。
- T01：analysis harness 自身不会把 TF32 全局状态污染留在进程中。

这些正向证据只缩小修复范围，不能覆盖同一 issue 下仍被黑盒复现的 residual。

## 下一步

进入 `RC4.5.2 checkpoint transactionality contract`，只冻结 B7/B24/B27 的统一合同：

1. current schema 的未知键策略必须明确，默认 fail-closed；
2. integrity 与 adversarial authenticity 的公开边界必须明确区分；
3. 所有 public filesystem errors 进入稳定 checkpoint exception hierarchy；
4. 所有 writer 采用相同 staging、atomic publication、caller-umask 与 cleanup 语义；
5. 所有身份和 payload 检查必须发生在 target mutation 前。

RC4.5.2 仍是 contract 阶段，不需要 H100，也不授权生产默认值改变。
