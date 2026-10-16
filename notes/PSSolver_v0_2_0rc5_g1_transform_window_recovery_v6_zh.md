# PSSolver rc5 G-1 transform measurement-window analysis-only recovery v6

本轮只修正资格分析器对 transform 计数窗口的解释，不修改求解器、profile 数据、科学阈值或生产默认值，也不提交 H100。

冻结的 96 份 H100 profile 显示，只有 `channel_legacy` 和 `channel_compiled_v2` 的 24 份记录为 `663 = 11 × 60 + 3`。这 3 次调用对应每 20 步一次、落入 60 步计时窗口的 scheduled spectral refresh；inverse 仍精确为 `36 × 60 = 2160`。其余 72 份记录没有窗口额外调用。

v2 合同分别保存并验证 raw count、scheduled refresh count 与 steady-state per-step count。`forward_per_step` 仍必须精确等于 raw count 除以 timed steps，原始 JSON 不做归一化或改写；分析器随后从 raw count 中扣除合同明确的 scheduled refresh calls，并要求稳态计数逐项精确匹配。`+2`、`+4`、inverse extra 或 raw/rate 不一致均 fail closed。

同时，versioned plan 改为绑定正式 H100 证据实际使用的 Provider runtime commit `dabeb9a487a82700ddd364c4233941a175808a3a` 与 wheel SHA-256 `954072040dc1681529c29e09d80f2fe9112d73442b98932978e1739c90212b22`。历史冻结记录保持原样，通过本记录的 forward-only successor 链解析。

本地完整 CPU suite 为 `3730 passed, 1 skipped, 7 deselected, 8 subtests passed`；唯一 skip 是既有 optional `nematics3d` dependency skip。`nematics3d` 未被安装或修改。

本记录尚不是累计资格 PASS。下一门禁是对 recovery-v5 中 96 份不可变 H100 profile 和 auxiliary evidence 的 CPU-only analyzer replay。
