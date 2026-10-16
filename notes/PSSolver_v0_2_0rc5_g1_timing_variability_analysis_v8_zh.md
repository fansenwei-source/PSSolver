# PSSolver v0.2.0rc5 G-1 timing variability analysis v8

## 结论

本轮是纯 analysis-only 诊断，没有提交 H100 Job，也没有重新裁决 v7。

最终分类：

`HARNESS_FIRST_SAMPLE_CONTAMINATION_AND_RESIDUAL_IN_WINDOW_VARIABILITY`

v7 的正式分类保持：

`FAIL_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_GATE`

## 输入和可复现性

- v7 profiles：48/48，只读；
- v7 控制目录 checksum：36/36 PASS；
- v7 scratch checksum：1119/1119 PASS；
- analyzer commit：`5a64129db775ed476997d17d291493105594a0cc`；
- analyzer tests：5 passed；
- 原始 timing 样本没有修改；
- 没有删除 outlier；
- 没有替换 trial；
- 没有修改任何资格阈值。

## 第一层原因：计量窗口首样本污染

版本化 AST 审计确认，`profile_rc458_runtime_family.run_profile()` 的正式顺序为：

1. `gc.collect()`；
2. `torch.cuda.empty_cache()`；
3. `torch.cuda.reset_peak_memory_stats()`；
4. `timer.reset()`；
5. 立即进入 `_timed_steps()`。

`empty_cache()` 与正式计时之间没有 untimed rewarm，所以释放 allocator cache 后的第一步被计入正式窗口。

48 份 profiles 中有 37 份的第一样本超过本 profile median 的 1.10 倍：

- Plane compiled v2：12/12；
- Channel compiled v2：12/12；
- Periodic functional：12/12；
- Channel functional：1/12。

因此，首样本 allocator recovery 污染得到证据支持。

## 第二层原因：首样本不是完整解释

冻结 CV 门限为 0.05：

- 原始正式窗口：32/48 profiles 超限；
- 仅作诊断、排除第一样本后：仍有 21/48 profiles 超限。

分 family 结果：

| Family | 正式 CV 失败 | 诊断性去首样本后仍失败 | 第一项开销 | 诊断 spike 数 |
|---|---:|---:|---:|---:|
| Plane compiled v2 | 12/12 | 3/12 | 12/12 | 34 |
| Channel compiled v2 | 7/12 | 7/12 | 12/12 | 35 |
| Periodic functional | 3/12 | 2/12 | 12/12 | 27 |
| Channel functional | 10/12 | 9/12 | 1/12 | 36 |

因此，简单增加一个 untimed rewarm 虽然必要，但不足以自动恢复资格。

## 配对顺序观察

四个 cell 均观察到 candidate-first 三个配对的平均 ratio 高于 baseline-first 三个配对：

| Family | baseline-first ratio mean | candidate-first ratio mean | 去首样本后配对相关性 median |
|---|---:|---:|---:|
| Plane compiled v2 | 0.995306403 | 1.005632486 | 0.190396195 |
| Channel compiled v2 | 0.987463148 | 1.003905154 | 0.111318553 |
| Periodic functional | 1.002317551 | 1.019232701 | 0.185141985 |
| Channel functional | 1.002475317 | 1.004400396 | -0.064057838 |

这一方向一致的顺序敏感性和较低的去首样本配对相关性支持“存在窗口内随机或环境波动”，但每个方向只有三个配对，不能据此唯一归因于 GPU clocks、节点竞争、allocator、pressure iterations 或某个求解器组件。

## 下一步合同建议

下一阶段应先预注册 measurement-window correction，不应直接重跑同一合同：

1. 将 post-`empty_cache` allocation recovery 与正式 timing window 分离；
2. timing 和 peak-memory 的计量阶段明确分层；
3. 保留全部 raw samples，禁止删除 outlier 和替换 trial；
4. 增加逐步 workload/pressure-iteration 与设备 clock/power provenance，用于解析剩余 21 项波动；
5. 在新的 H100 submission 之前，以 synthetic tests 固定阶段顺序、状态步数和 transform 窗口；
6. 不因本次 analysis-only 结果放宽 1.03、1.05 或 CV 门限。

当前：

- `analysis_complete = true`
- `g1_complete = false`
- `eligible_for_cumulative_reassembly = false`
- `eligible_for_automatic_merge = false`
- `eligible_for_default_promotion = false`
- `new_h100_submission_count = 0`
- `production_default_changed = false`
