# PSSolver v0.3.0 S-09 6×60 失败原因边界分析

分类：`TRANSIENT_PROFILE_SLOWDOWN_SUPPORTED_CAUSE_UNRESOLVED`

本分析只读取已经完成且内容寻址的 6×60 自对照、候选矩阵，以及此前 5×30 的接受前真实残差重算干预矩阵；没有执行新的 performance profile，失败资格判定保持不变。

失败的 Channel functional tolerance trial 1 中，baseline 和 candidate 最终都使用 27 次压力迭代，因此不能用额外迭代数解释变慢。候选前 30 步相对 baseline 慢 `12.2236%`，后 30 步慢 `2.2720%`。候选前三个十步块均值严格下降，而其余五个 trial 的 profile-center 配对比均不超过 `1.031599`。

此前移除接受前真实残差重算的干预实验显示，这项正确性检查在 functional tolerance 路径上的中位成本约为 `0.3577%`，最大配对成本约为 `2.1386%`。它支持一个小而稳定的真实成本，但不足以解释本次 `7.0771%` 失败。

失败也不能只归因于单个 `0.265121` 秒样本，因为其后的两个固定块仍明显偏慢。现有证据支持“单 profile 前半段暂态变慢”，但不能唯一辨别调度、CPU 频率、cache 或 runtime warmup。若要继续区分，需要用户重新授权一次预登记、带系统遥测的定向测量。

S-09 仍未完成，R-0022 未创建，H100 准备不允许。本记录不授权任何远端或外部操作。
