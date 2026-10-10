# PSSolver RC4.2.6 跨设备 checkpoint H100 关闭记录

RC4.2 已完成，B10、B22、N3、N5 和 N6 从 `partial_residual` 转为 `closed_verified`。权威分类为：

`PASS_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY`

## 已关闭的合同

RC4.2 把 checkpoint 身份拆成科学、离散、动力学、状态布局、backend restart、lifting 和 derivative 层。设备请求 token 与 fresh-only 初值 conditioning 不再错误地进入兼容身份；Channel Nyquist、Periodic Hermitian repair、lifting dynamics 与 functional derivative contract 均由显式版本标识。旧格式只能通过已登记 reader 接受或在 target mutation 前明确拒绝。

## CPU 与 installed-wheel 证据

RC4.2.5 已完成 source、installed-wheel、legacy acceptance/rejection、restart equivalence 和 pre-mutation negative gates。RC4.2.6 v9 复用了 checksum 完整的 CPU 证据，并在没有 source shadow import 的 fresh wheel 环境中运行资格 harness，结果为 `26 passed`。

## H100 结果

唯一作业 `10874642` 在 `gpu-h100-4-0` 上以 `COMPLETED / 0:0` 结束，用时 35 秒。P01–P08 preflight 与 7 个 CUDA-only tests 全部通过，TF32 在 Python 进程内显式关闭。

X01–X14 覆盖 Plane 三条 runtime、Periodic production、Channel production，以及 Periodic/Channel functional runtime 的 CPU→CUDA 和 CUDA→CPU：

- 14/14 compatibility identity 相同；
- 14/14 serialized payload、progress 与 persistent backend state 精确恢复；
- 14/14 恢复状态与恢复后一步有限；
- 14/14 raw-byte tamper 在 target mutation 前被拒绝；
- 14/14 source checkpoint tree 未改变；
- runtime fallback 为 0/14。

本结论只覆盖 checkpoint 精确物化、可用恢复和完整性保护，不声称跨设备继续演化逐字节一致、长期科学等价或性能/显存非回退。

## 归档

控制目录：

`/home/fansenwei/pssolver_rc426_cross_device_checkpoint_h100_5e3e821_20261010_recovery_v9`

控制 manifest：1428/1428 PASS，SHA-256：

`5c4053ce0807bd63c3b2016b9fc6bf69fd4410a08ed57af6b97f528cfef9d147`

Scratch manifest：673/673 PASS，SHA-256：

`3ead63c184979cff688fca9c5b2ae3a08ee8782f7a7a1434341b0730269469d4`

`COMPLETE` 已生成。worktree 保持 detached、clean 且无缓存。

## 下一步边界

只授权进入 RC4.3 lifting correctness 的规划阶段。当前不授权修改 lifting 数学、自动 merge、默认值提升或发布；PSSolver-Control 与 `nematics3d` 均不需要因本关闭记录而修改。
