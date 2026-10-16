# PSSolver RC5 G-1 v13 paired-block 失败分析

## 结论

本次只做 analysis-only 诊断，不重新裁决正式资格结果。

- 正式 v13 分类继续保持为
  `FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE`；
- 没有授予 PASS；
- `qualification_complete = false`；
- `g1_complete = false`；
- 没有提交新的 H100 Job；
- 没有修改阈值、删除样本、替换 trial 或修改生产默认值。

诊断分类为：

`ANALYSIS_ONLY_V13_MAX_TRIAL_FAILURE_CELL_MIGRATION_SUPPORTED`

它表示：现有证据支持“maximum-per-trial 门禁存在跨运行重复性问题”，但不支持把
v13 的失败直接解释为 candidate 持续变慢，也尚未找到唯一的环境根因。

## 冻结证据完整性

HPCC 登录节点重新只读验证了四套证据：

- v10：41/41 PASS；
- v11：8/8 PASS；
- v12：83/83 PASS；
- v13：1803/1803 PASS。

旧归档均未修改。

## 跨运行失败单元迁移

v10 的 paired-block post-hoc replay 只在以下单元失败：

`channel_compiled_v2/channel_R128x20x20`

其 maximum trial median fixed-block ratio 为：

`1.0642916340012079`

同一套 v10 证据中，Periodic R128 的对应最大值只有：

`1.0109350621196627`

正式 v13 则只在以下单元失败：

`periodic_functional/periodic_R128`

其最大值为：

`1.0633863358803155`

也就是说，失败位置从 v10 的 Channel 迁移到了 v13 的 Periodic。两次运行中，所有
aggregate、group median 和 memory 门禁都通过；两次失败都只来自
`every_trial_median_fixed_block_ratio <= 1.05`。

## v13 Periodic 原始结构

trial 3 的六个 block ratio 是：

```text
1.0659403372690608
1.0642290123274774
1.0625436594331534
1.0683268119237934
0.8778866894127096
0.994775914093373
```

前四个 block 超过 1.05，因此 trial median 为 1.0633863358803155。

但同一次 Job 的 trial 5 中，baseline 和 candidate 各自的 CV 都不超过 0.05，六个
block ratio 却全部低于 0.95：

```text
0.9326291223116029
0.9280643385705132
0.9336282904749439
0.9353811745866526
0.9362949098760027
0.936059976952602
```

六个 trial median 的范围从 0.9345047325307982 到 1.0633863358803155，跨度为
0.12888160334951726。奇数 trial 的执行顺序是 baseline→candidate，偶数 trial 是
candidate→baseline；两组 trial median 的中位数分别为 0.9977281144443686 和
1.0017715403077303，因此没有观察到一个简单、固定的先后顺序偏差。

## 为什么不能直接再跑一次

当前 block 是按两个先后执行的独立 profile 中相同的样本下标配对，并不是在同一时间
窗口同步测量。若节点性能状态在两个 profile 之间改变，相同 block 编号不能消除这种
漂移。再加上门禁取六个 trial 中的最大值，单个局部状态变化即可否决整个 cell。

v11 已记录 position-drifting timing variability；v12 又排除了 GC、scheduled refresh
或 pressure termination 作为唯一主因，但没有找到唯一替代原因。因此，原样重试 v13
既不能解决实验设计问题，也可能只是把失败迁移到另一个 cell。

## 下一步

在授权新的 H100 qualification 之前，应先评审实验配对方式和 maximum-per-trial 门禁。
这一步需要形成新的、预注册的统计合同；不能在看到本次结果后临时放宽 1.05 阈值，
也不能删除不利样本或替换 trial。

analysis-only 归档：

`/home/fansenwei/pssolver_rc5_g1_paired_block_failure_analysis_dabeb9a_20261016_v14`

归档 checksum：12/12 PASS；`checksums.sha256` 自身 SHA-256 为：

`7df3c1685cb893379506a932a52b5371f315cf11214a03f38793a25c77b4bdee`
