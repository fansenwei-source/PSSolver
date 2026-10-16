# PSSolver rc5 G-1 paired-block measurement contract v13

状态：`READY_FOR_G1_PAIRED_BLOCK_H100_QUALIFICATION`

v13 estimator 已获得正式采用、push 和单次 H100 qualification 授权；自动重试仍被禁止。这个状态不是 G-1 qualification PASS，历史 replay 也不能替代新的正式 H100 证据。

## 为什么需要新合同

v12 的 12 个 instrumented profiles 表明 timing spikes 的位置随 trial 和窗口漂移；GC、scheduled refresh、pressure termination 和逐步 synchronization 都不能解释为单一主因。继续把每个 60-step profile 的 `CV <= 0.05` 当作候选实现的独立否决条件，会把环境抖动直接归因给候选代码，现有证据不支持这种归因。

## 正式采用的测量语义

每个 profile 的 60 个原始样本全部保留，并按原顺序固定划分为六个连续的 10-sample blocks。每个 block 使用算术平均；同一 trial 的 candidate/baseline 按 block index 形成六个 ratio，并使用其 median 作为 trial ratio。四个 focused cells 各保留六个 AB/BA order-balanced trials。

正式数值界限不降低：

- 所有 360 个 candidate/baseline 样本的 aggregate mean ratio：`<= 1.03`；
- 六个 profile means 的 median ratio：`<= 1.03`；
- 36 个 fixed-block ratios 的 median：`<= 1.03`；
- 每个 trial 的 median fixed-block ratio：`<= 1.05`；
- peak allocated / reserved group ratio：`<= 1.03`。

`CV = 0.05` 仍逐 profile 重算并报告，但在 v13 合同中只作为环境诊断，不再单独否决候选。这个变化不是阈值放宽，而是 estimator/attribution 语义变化；用户已于 2026-10-16 明确授权正式采用。

禁止删除 sample、删除首样本、重排 block、替换 trial 或事后选择窗口。缺失、非有限、summary 与 raw sample 不一致、profile/progress 文件集合错误、执行顺序或文件 SHA 不一致均 fail closed。

## v10 历史数据只读 replay

48 个 profile 和 48 个累积 progress records 均通过文件集合、顺序及 SHA 验证。replay 分类为：

`ANALYSIS_ONLY_V10_REPLAY_FAILS_PROPOSED_V13_CONTRACT`

结果：

- `plane_compiled_v2 / plane_R128`：PASS，最大 trial block-median ratio `1.0244558274`；
- `periodic_functional / periodic_R128`：PASS，最大值 `1.0109350621`；
- `channel_functional / channel_R512x40x40`：PASS，最大值 `1.0145428433`；
- `channel_compiled_v2 / channel_R128x20x20`：FAIL，最大值 `1.0642916340 > 1.05`。

因此旧 v10 数据没有被重新解释为通过，v10 的正式 FAIL 保持不变。这个结果也说明 proposed estimator 没有把全部高-CV 数据无条件放行：它仍能拒绝局部、成对且超过 1.05 的候选回退。

## CPU synthetic/negative controls

- baseline/candidate 共同经历相同稀疏 spikes、真实 ratio 为 1.02：通过；全部 60 个样本仍参与计算；
- candidate ratio 为 1.031：拒绝；
- candidate ratio 为 1.06：拒绝；
- 缺失、NaN/Inf、progress 顺序或 SHA tamper：拒绝。

## 后续门禁

下一门禁是 push 精确版本化提交，然后用绑定 plan/analyzer SHA-256 的外部授权记录执行一次新的、预注册的 48-profile paired-block measurement run。不得用本次 post-hoc replay 授予资格，也不得自动重试。

qualification-support runner 默认只能生成和审计 48 项 command plan。进入执行模式必须提供一个普通文件、非 symlink 的外部授权记录；该记录必须同时：绑定 versioned plan/analyzer SHA-256、明确采用 v13 合同、只授权一次 H100 submission，并规定 `automatic_retry=false`。缺失、篡改或范围不符时会在任何 profile output 创建前拒绝执行。

qualification-support 的 HPCC 登录节点 CPU 验证结果为：相关定向集合 `28 passed`，全部 RC5 G-1 测试 `129 passed`，均为 `0 failed / 0 skipped`。环境为 Python 3.10.18、pytest 9.1.1；这些结果不包含且不替代 H100 资格证据。
