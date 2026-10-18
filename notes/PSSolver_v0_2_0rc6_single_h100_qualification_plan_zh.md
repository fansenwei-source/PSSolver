# PSSolver v0.2.0rc6 单次 H100 累计资格方案

状态：`READY_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION_SUPPORT`

本阶段只冻结合同，没有 push，也没有提交 H100 Job。

## 固定对象

- Provider baseline：v0.2.0rc5，`674a812...`，wheel SHA-256 `3097b08b...`
- Provider candidate：v0.2.0rc6，`ea801ec...`，wheel SHA-256 `8840ebfa...`
- Consumer candidate：`208edd0...`，wheel SHA-256 `eca0c2c1...`
- functional API：1.0
- RC6 Provider identity 相对 RC5 不变，不需要新的 Consumer identity migration。

## 唯一作业顺序

1. node-local `/tmp`、`RENAME_NOREPLACE`、环境、compiler 和空 cache compile preflight；
2. 7 个 Provider CUDA-only tests；
3. RC6-1 至 RC6-6 中适用于设备执行的错误、pressure、checkpoint 和 restart smoke；
4. 6 caller states × 8 runtime families 的 TF32 restoration matrix；
5. 四类当前 checkpoint 的 CPU→CUDA 与 CUDA→CPU transactional restart；
6. rc1–rc5 已发布 checkpoint disposition matrix；
7. 8 个 runtime family 的短程数值门禁；
8. 全部 16 个 runtime/grid cell 的 balanced-crossover 性能与显存矩阵；
9. installed-wheel PSSolver-Control 完整 H100 suite 与 guard-order 门禁；
10. 累计 analyzer、checksum 复核，全部通过后才允许写 `COMPLETE`。

任一门禁失败都立即停止；禁止自动重试，也禁止未经新授权提交第二个 Job。

## 性能合同

性能矩阵覆盖 8 个 runtime family、每类 2 个网格、每 cell 6 个 trial、每 trial 两个
Provider 角色，共 192 个 profile。奇数 trial 为 baseline→candidate，偶数 trial 为
candidate→baseline。每个 profile 保留全部 60 个样本，分成六个固定 10-step block，
profile center 为六个 block mean 的中位数。

采用 RC5 已正式批准的阈值：总体与 median ratio 不超过 1.03；两个顺序 strata 与顺序
imbalance 不超过 1.05；至少 4/6 trial ratio 不超过 1.05；peak allocated/reserved ratio
不超过 1.03。不得删除样本、剔除 outlier 或替换 trial。

## Consumer 合同

H100 上预计 `344 passed, 1 skipped`。唯一允许 skip 是 opt-in R5 benchmark；不允许 failure、
xfail 或 deselect。必须验证 guard-order regression、Periodic/Channel CUDA finite execution、
checkpoint replay 和 gradient gates。

## 成功语义与边界

成功分类为 `PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION`。成功只使 RC6 具备 release
integration review 资格；不自动 merge、tag、发布、修改默认值或声明科学 control 结果，
并且发布后仍需独立验收。

下一步是在资格专用文件范围内实现 runner、analyzer 和 Slurm script，并完成 CPU synthetic
与 planning preflight。该工作不授权 push 或 H100 submission。
