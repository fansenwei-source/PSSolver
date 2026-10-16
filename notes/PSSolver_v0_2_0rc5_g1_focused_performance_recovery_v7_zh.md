# PSSolver rc5 G-1 focused performance/memory recovery v7 预注册

本阶段只准备新的 focused H100 证据合同，不提交 H100、不 push，也不修改求解器、生产默认值或既有科学阈值。v6 的失败分类保持不变。

## 为什么不能直接把 v6 判成 timing noise

四个失败 cell 并不属于同一种现象：

- `plane_compiled_v2 / plane_R128` 和 `periodic_functional / periodic_R128` 的失败 trial 中，candidate 的 60-step 样本出现明显尖峰，均值被显著抬高；
- `channel_compiled_v2 / channel_R128x20x20` 的均值和中位数都稳定地慢约 5.9%，因此真实回退仍然成立，不能用“噪声”覆盖；
- `channel_functional / channel_R512x40x40` 的一个 candidate trial 比其余 trial 多保留整整 64 MiB。虽然 peak allocated 和时间门禁通过，但原有 group-max reserved-memory 门禁仍然失败。

所以本 recovery 不删除异常值、不改用更宽松的多数票，也不降低阈值，而是重新收集一组事先冻结、执行顺序平衡的新证据。

## 新 focused matrix

只运行上述 4 个失败 cell。每个 cell 运行 6 个 matched trials，每个 trial 都包含 baseline 和 candidate，共 48 个 profile。奇数 trial 按 baseline→candidate 执行，偶数 trial 按 candidate→baseline 执行；每个 cell 恰有三组 baseline-first 和三组 candidate-first，避免单向的热机或节点漂移偏置。

每份 profile 仍使用 10 个 warmup steps 和 60 个正式计时 steps，TF32 关闭，并使用独立且初始为空的 compiler cache。

## 计时有效性先决门禁

Analyzer 从 60 个原始样本重新计算 mean、median、sample standard deviation 和 coefficient of variation，并要求持久化 summary 与重算结果一致。每份 profile 都必须满足 `CV <= 0.05`。

如果任何 profile 的 CV 超限，整批证据立即停止；不允许删除 outlier、不允许补跑或替换某个 trial，也不能把其余 trial 选出来继续裁决。这一门禁只判断测量是否有效，不改变性能阈值。

## 保持不变的性能与显存门禁

- group mean candidate/baseline ≤ 1.03；
- group median-of-profile-means candidate/baseline ≤ 1.03；
- 六个 paired mean ratio 必须每个都 ≤ 1.05；
- group-max peak allocated ratio ≤ 1.03；
- group-max peak reserved ratio ≤ 1.03。

runtime fallback、compile fallback、graph break、初态身份、transform raw/steady count 和非有限状态仍按 fail-closed 方式检查。对于未涉及 periodic repair 的 family，A/B final-state SHA 仍须一致。

## 裁决边界

focused analyzer 即使 PASS，也只产生 `PASS_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_EVIDENCE`，并设置 `eligible_for_cumulative_reassembly=true`；它不会直接声明 `qualification_complete` 或 `g1_complete`。随后还必须重新验证 v5/v6 checksum，并由单独的累计 analyzer 将 12 个原已通过 cell 与 4 个新 focused cell 重组后，才可能关闭 G-1。

用户随后已经明确授权 push `fix/v0.2.0rc5` 和一次 G-1 focused performance/memory recovery v7 H100 submission，并继续禁止自动重试。版本化 runner 只接受 48 个精确 profile，使用独立 compiler cache，逐份持久化 progress；分析失败时写入 `ANALYSIS_FAILED.json`。Slurm 成功时只写 `FOCUSED_COMPLETE`，不会写代表整个 G-1 关闭的 `COMPLETE`。
