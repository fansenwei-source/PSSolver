# PSSolver v0.2.0rc5 G-1 v14 平衡交叉估计量方案

## 结论

本轮完成了 CPU-only 合同设计、合成负向控制，以及对冻结 v10/v13 原始 profile/progress 的 HPCC 登录节点只读回放。

分类为：

`ANALYSIS_ONLY_V10_V13_REPLAY_PASSES_PROPOSED_V14_BALANCED_CROSSOVER_CONTRACT`

这不等于 G-1 PASS。v10 与 v13 的正式 FAIL 分类均保持不变；没有提交 Slurm/H100 作业，没有修改求解器、runtime、科学方程或生产默认值。

## 为什么需要调整估计量

v13 合同把不同进程、先后执行的 baseline/candidate profile 中相同 block 序号直接配对，并允许六个 trial 中任意一个 trial 的 block-ratio 中位数否决整个 cell。v10 与 v13 的失败 cell 在 Channel compiled R128 和 Periodic R128 之间迁移，而两轮的总体、中心和显存门禁全部通过。这支持“单个 trial 最大值门禁不可重复”的担忧，不支持稳定的 provider 性能回退。

## v14 提议

保留：

- 4 个 cell、每 cell 6 个 trial、总计 48 份 profile；
- 每份 profile 的 60 个原始样本；
- 六个连续且不可重排的 10-sample block；
- 既有 1.03 总体门限、1.05 paired 参考门限和 1.03 显存门限；
- trial 1/3/5 为 baseline→candidate，trial 2/4/6 为 candidate→baseline。

估计方法：

1. 每个 block 取 10 个原始样本的算术平均；
2. 每份 profile 以六个 block 均值的中位数作为 profile center；
3. 每个 trial 计算 candidate center / baseline center；
4. 六个 trial 的中位数必须不超过 1.03；
5. 两个执行顺序层各自三个 trial 的中位数必须不超过 1.05；
6. 两个顺序层中位数的 max/min 必须不超过 1.05；
7. 六个 trial 至少 4 个不超过 1.05；
8. 原有总体均值、profile mean 中位数和显存门禁继续生效。

单个 trial 不再作为独立否决项，但其全部样本和结果仍保留并参与重复性、顺序层和总体裁决。

## 冻结证据回放

v10 四个 cell 的六-trial 中位数分别为：

- Plane compiled R128：1.007465028524；
- Channel compiled R128：1.003948195799，5/6 trial 不超过 1.05；
- Periodic R128：0.990521450489；
- Channel functional R512：1.007450939812。

v13 四个 cell 的六-trial 中位数分别为：

- Plane compiled R128：1.001610699582；
- Channel compiled R128：0.992857363331；
- Periodic R128：0.999877802612，5/6 trial 不超过 1.05；
- Channel functional R512：1.001097156623。

两轮全部顺序层中位数不超过 1.05，最大顺序层不平衡为 v10 Periodic 的 1.032023539632；总体性能和显存门禁也全部通过。

## 负向控制

以下情况均被拒绝：

- 六个 trial 全部为 1.031；
- 六个 trial 全部为 1.06；
- 奇数 trial 为 1.06、偶数 trial 为 0.95 的顺序偏差；
- 只有 3/6 trial 不超过 1.05；
- peak allocated 或 reserved ratio 为 1.031；
- NaN、Inf、零或负 ratio。

## HPCC 归档

控制目录：

`/home/fansenwei/pssolver_rc5_g1_balanced_crossover_replay_dabeb9a_20261016_v15`

- v10：41/41 checksum PASS；
- v13：1803/1803 checksum PASS；
- 新归档：19/19 checksum PASS；
- `checksums.sha256` SHA-256：`bb139129d5980d9d0fae95ae00275464d7cb0c9551ec49937c2a288a98a9c073`；
- H100/Slurm submission：0。

## 下一步边界

当前只允许评审该合同。只有用户明确授权正式采用后，才能实现 source-bound runner、push 分支，并申请一次新的 single-H100 qualification。现有回放不能追溯把 v10 或 v13 改判为 PASS。
