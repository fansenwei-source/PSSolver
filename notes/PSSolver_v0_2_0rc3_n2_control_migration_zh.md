# PSSolver v0.2.0rc3 N2：PSSolver-Control 显式声明迁移

## 结论

PSSolver 保留 rc2 引入的显式公共构造器合同，不恢复可能跨应用产生歧义的隐式默认值：

- `SpectralNumerics.spectral_storage` 必填；
- `Output.save_start_step` 必填；
- `Output.diagnostics` 必填；
- `Output.save_hydrodynamics` 必填。

这是相对于 rc1 的 source-level breaking change，但不是生产 runtime 默认选择的改变。
PSSolver 的 CHANGELOG 和架构示例已经补充 breaking 标记与迁移示例。

## 所有权边界

PSSolver-Control 由其独立维护者（Claude Code）负责。本提交不修改、reset、stash、clean、
commit 或 push PSSolver-Control，也不修改 nematics3d。Control 迁移应在其当前工作分支的最
新状态上完成，不得覆盖尚未推送的研究提交。

## 2026-10-06 只读审计结果

以下五处 `SpectralNumerics` 已经显式声明了正确的存储方式，不需要再改：

- `tests/periodic_runtime.py`：`hermitian_half`，`hermitian_axis=1`；
- `tests/channel_runtime.py`：`full_complex`；
- `experiments/warmup_3d/periodic_box.py`：`hermitian_half`；
- `experiments/warmup_3d/hidden_mode_reproducer.py`：`hermitian_half`；
- `experiments/warmup_3d/r1_bridge.py`：`hermitian_half`。

上述五处 `Output(...)` 仍缺少三个必填策略。为保持这些调用在 rc1 中实际获得的旧值，应在
每一处加入：

```python
save_start_step=0,
diagnostics=True,
save_hydrodynamics=True,
```

精确文件为：

1. `tests/periodic_runtime.py`；
2. `tests/channel_runtime.py`；
3. `experiments/warmup_3d/periodic_box.py`；
4. `experiments/warmup_3d/hidden_mode_reproducer.py`；
5. `experiments/warmup_3d/r1_bridge.py`。

## Consumer 资格要求

Control 维护者应在不修改 PSSolver Provider 的前提下完成以下门禁：

1. 五个调用点都显式给出三个 Output 策略；
2. 不恢复 Consumer 自己的隐式 wrapper 默认；
3. 使用包含 N1 修复的 Provider commit
   `165d3b6a4007f8ebd46f3455ceb2a4d57667653c`；
4. 两个 runtime construction tests 必须通过；
5. 三个 warm-up 构造函数至少各完成一次 import/compile smoke；
6. Control 完整 CPU suite 不得出现新增 failure、skip 或 xfail；
7. 证明迁移前后生成的 canonical simulation metadata 除新增显式字段的来源说明外保持一致；
8. 不宣称修改了物理、谱方法、边界条件、优化算法或生产默认值。

Control 完成并推送自己的迁移提交后，PSSolver rc3 只记录 Consumer commit 身份和测试结果，
不把 Consumer 源码复制进 Provider 仓库。

## Provider 本构兼容记录

后续只读审计发现，Control planar adapter 的 whole-file constitutive pin 会被 rc3 Provider
中与所消费 helper 无关的 runtime-adapter 改动触发。Provider 已将允许更新该 pin 所需的
符号级证据单独记录在：

- `notes/PSSolver_v0_2_0rc3_constitutive_compatibility.json`；
- `notes/PSSolver_v0_2_0rc3_constitutive_compatibility_zh.md`。

该记录只覆盖 Control 明确消费的六个 helper 与 `q_tensor.py`，不宣称整个
`beris_edwards.py` 等价。Control 必须在独立提交中更新 pin/provenance，并在未绕过 pin
的情况下重新运行自己的资格门禁。
