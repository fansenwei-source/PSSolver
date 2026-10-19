# PSSolver v0.2.0rc6 累计 H100 资格关闭记录

## 结论

RC6 的单次 H100 累计资格验证已经通过，正式分类为：

`PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION`

通过资格验证的运行时源码仍是 `ea801ec08f5b2465661334e1c126133e55b68e96`；执行资格编排的 support commit 是 `3228d3d723393818ee6149e9cf56021ffaaed411`。本记录是后续证据登记，不把记录提交本身描述为已通过 H100 的源码。

## H100 作业

- Job ID：`10901236`
- 节点：`gpu-h100-4-0`
- 状态 / ExitCode：`COMPLETED / 0:0`
- 用时：`00:41:31`
- recovery v7 submission：1 次
- 自动重试：false
- 第二个 v7 Job：未提交

## 累计门禁

冻结的 11 个阶段全部通过，包括环境、7 个 CUDA-only tests、公共合同 smoke、TF32 scope、当前与历史 checkpoint、8 个 runtime family 的数值身份、192 项 profile、Consumer、辅助汇总和最终 analyzer。

192 份 profile 与 192 份 profile-child TF32 bootstrap 均完整。每个子进程都在 profile 启动前原子持久化并验证 `highest / CUDA matmul false / cuDNN false`；每份 profile 也独立报告相同的运行时状态。全部 profile finite，graph break 为 0，没有 compile 或 runtime fallback。

16/16 性能与显存分组通过全部预注册门禁。最大 aggregate all-sample mean ratio 为 1.0114777680647684，最大 median-of-six ratio 为 1.0066664043431481，最大 order-stratum imbalance ratio 为 1.0229522142590188，peak allocated/reserved group ratio 均为 1.0。

Consumer 门禁为 344 passed、1 个既有 R5 benchmark skip、0 failed。Periodic 与 Channel CUDA 执行、checkpoint replay、gradient 和 guard-order regression 均通过。

## 归档

控制目录为：

`/home/fansenwei/pssolver_rc6_h100_ea801ec_208edd0_20261019_recovery_v7`

正式输出目录为：

`/scratch1/vincent/PSSolver/data_pssolver_rc6_h100_ea801ec_208edd0_20261019_recovery_v7`

两侧 `COMPLETE` 均存在。控制目录 checksum 为 24/24 PASS，manifest SHA-256 为 `26c99535fc92b1c33d0a8fe6adc5c2fab05d0332034eb7464657a4e047ecbb40`；正式输出 checksum 为 4763/4763 PASS，manifest SHA-256 为 `160a4960da3de7ea518ff2187393a6d7bb6e563a9611be1057ecb6c437ad894c`。

仓库登记门禁为 28 个定向测试通过，以及完整 CPU suite `4062 passed, 7 deselected, 8 subtests passed`；没有 failure、skip 或 xfail。七个 deselect 精确对应既有 CUDA-only node ID。

## 边界

`rc6_complete = true`，并且可以进入 release-integration review。但这不授权自动合并、默认 runtime promotion、tag、GitHub Release 或 PyPI 发布；独立 post-release verification 仍然是必需门禁。PSSolver-Control、nematics3d、生产默认值和数值门限均未修改。
