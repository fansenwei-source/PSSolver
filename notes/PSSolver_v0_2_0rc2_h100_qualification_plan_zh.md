# PSSolver v0.2.0rc2 审计修复 H100 资格计划

## 1. 验证对象

本计划把运行时代码冻结在：

- A：`v0.2.0rc1` / `1c5237194c5e4f696bb5d3d01d2caa2c4bdd27a1`
- B：`fix/v0.2.0rc2-audit` 的运行时提交 `1c8b3330139dec7cbb2ea138b261fcaadf5e871f`

本计划文件之后产生的提交只用于携带资格合同，不改变 B 的运行时代码。正式 H100 工作树必须 checkout 上述两个精确提交，不能把分支名当作可漂移的验证对象。

设备字段采用两阶段合同：调用 `profile_periodic_hermitian_qualification.py` 时传入的 CLI token 必须是 `cuda`；profiler 分配当前可见 GPU 后，正式 JSON 中的 `config.device` 和 `environment.device` 必须规范化为 `cuda:0`。不得把输出身份 `cuda:0` 原样传回只接受 `cpu`/`cuda` 的基础 profiler。

## 2. 为什么不能只跑一次 smoke

本轮同时改变了 Periodic Hermitian 投影、Channel dealias 后投影、Nyquist 处理、lifting、TF32 policy 和极端尺度 pressure adjoint。短 smoke 可以发现 import、CUDA 和 finite 错误，但不能发现：

- 经过几十到几百时间单位才增长的 hidden anti-Hermitian mode；
- correct projection 引入的稳定性能或显存回退；
- Channel 速度在 dealias 后重新获得的纵向分量；
- checkpoint/restart 在 GPU 上的状态身份差异。

因此采用“一次 H100 Job、多个 fail-fast gate”，而不是为每个 bug 分别提交作业。

## 3. 执行顺序

1. 验证 Git wrapper、A/B 提交、父链、changed-file 范围和 `git diff --check`。
2. 建立两个全新 detached clean worktree。
3. 分别从 A/B 的 clean archive 构建 wheel，并安装到两个独立、继承固定 PyTorch 依赖的 venv；禁止 source shadow import。
4. 登录节点运行 B 完整 CPU suite，精确 deselect 六个 CUDA-only node ID；唯一允许的 skip 仍是 optional `nematics3d` dependency。
5. 创建全新 control 和 scratch 目录，提交且只提交一个 H100 Job。
6. H100 Job 内先运行六个 CUDA-only tests；任一失败立即停止。
7. 使用 `benchmarks/check_periodic_hermitian_stability.py` 对 `loop3d` 和 `r1`、`full_complex` 和 `hermitian_half` 四种组合分别跑到 `t=200`。
8. 使用 `benchmarks/profile_periodic_hermitian_qualification.py --device cuda` 生成 36 份 A/B profile；每份 JSON 必须记录实际分配身份 `cuda:0`，再用 `benchmarks/analyze_periodic_hermitian_profiles.py` 裁决。
9. 运行 machine-readable 合同 G4 中的 Nyquist、Channel、TF32、lifting、finite-Q 和 pressure gates。
10. 运行 R128 100-step production/restart smoke，完成 checksum、provenance、summary 和 final report。

## 4. 停止规则

以下任一情况立即停止，不得自动重试：

- 提交、worktree、wheel 或 import 身份不符；
- CPU 出现任何 failure、额外 skip、xfail 或额外 deselect；
- H100 型号、可见设备数量或 TF32 状态不符；
- 任一 long-horizon sample 非有限或超过 Hermitian bound；
- profile JSON 不完整、CV/性能/显存门禁失败；
- divergence、Nyquist、full/truncated、restart、pressure 或 checkpoint gate 失败；
- OOM、NaN、Inf、CUDA error、runtime fallback 或 source shadow import。

外部 qualification helper 自身失败时必须单独标为 harness failure，不能冒充 solver/scientific failure，但同样不得在没有新授权时提交第二个 Job。

## 5. 结论边界

全部 gate 通过时，只能给出：

```text
PASS_V0_2_0RC2_AUDIT_H100_NON_REGRESSION
eligible_for_rc2_integration_review = true
```

它不自动授权 merge、push、默认值晋升、正式发布或长期科学模拟。

本计划也不处理 R1–R9 结构冗余；结构清理必须在 rc2 行为资格关闭后另开分支。
