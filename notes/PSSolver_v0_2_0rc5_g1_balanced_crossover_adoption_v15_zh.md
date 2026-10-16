# PSSolver v0.2.0rc5 G-1 平衡交叉合同正式采用与执行准备

## 状态

`READY_FOR_G1_BALANCED_CROSSOVER_H100_AUTHORIZATION`

v14 提议的平衡交叉估计量现已在本地正式采用，并完成 source-bound qualification support。采用合同不等于通过资格验证，也不自动授权 push 或 H100。

## 正式采用的估计量

- 仍运行 4 个 cell、每个 cell 6 个交叉顺序 trial，共 48 份 profile；
- 每份 profile 保留全部 60 个 timing 样本；
- 每 10 个连续样本构成一个固定 block；
- 每份 profile 的中心是六个 block 算术均值的中位数；
- trial ratio 是 candidate profile center / baseline profile center；
- 六个 trial 的中位数必须不超过 1.03；
- baseline→candidate 与 candidate→baseline 两个顺序层的中位数分别不超过 1.05；
- 两个顺序层的 max/min 不超过 1.05；
- 至少 4/6 个 trial ratio 不超过 1.05；
- 原总体均值、profile mean 中位数与显存 1.03 门禁继续保留。

没有删除样本、移除 outlier、替换 trial 或放宽数值门限。变化仅限于不再让单个 trial 最大值独立否决整个 cell。

## 执行支持

新增了：

- 正式 qualification plan；
- 正式 analyzer；
- 48-profile versioned runner；
- single-H100 Slurm wrapper；
- source-bound authorization 和哈希篡改负向门禁。

runner 在没有外部授权记录时会拒绝执行。授权记录必须：

1. 是普通文件且不能是符号链接；
2. 明确授权正式合同和一次 H100 submission；
3. `automatic_retry=false`；
4. 精确绑定五个执行关键源文件的 SHA-256；
5. 任一源文件被修改后立即失效。

Slurm wrapper 继续要求 node-local `/tmp` 上的 `RENAME_NOREPLACE` preflight，并把 `TMPDIR/TMP/TEMP` 固定到独立目录。Job 只能写 `BALANCED_CROSSOVER_COMPLETE`，不能直接写累计资格的 `COMPLETE`。

## CPU 验证

- 合同与执行支持定向测试：22 passed；
- 全部 RC5 G-1 测试：206 passed；
- 使用 v13 的 48 份冻结 profile 和临时新 progress schema 做 formal analyzer smoke：4/4 cell PASS；
- 隔离 qualification-only support planning：48/48 commands，未创建 output root；
- 完整 checkout 被 support root 防护正确拒绝；
- Bash syntax：PASS。

上述 smoke 不是正式重新资格验证，不改变 v10 或 v13 的 FAIL 分类。

## 当前边界

- 没有 push；
- 没有调用 `sbatch`；
- H100 submission 数量为 0；
- 没有修改 production solver、runtime、科学方程或默认值；
- `qualification_complete=false`；
- `g1_complete=false`。

下一步需要用户明确授权：push `fix/v0.2.0rc5`，并允许一次 source-bound balanced-crossover single-H100 qualification submission；不得自动重试。
