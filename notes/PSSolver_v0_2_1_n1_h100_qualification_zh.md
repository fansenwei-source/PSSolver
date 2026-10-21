# PSSolver v0.2.1 N-1 累计资格记录

日期：2026-10-20

最终分类：`PASS_V0_2_1_N1_CUMULATIVE_QUALIFICATION`

真实 NFS CPU 资格验证之后，唯一 small-H100 Job `10902665` 已成功完成。运行时源码为
`15670eb264a44c6707aeb1695b74e6cdbb34f611`，N-1 实现提交为
`927dc89aa63bcee9b03248f52662389bc947c47a`；H100 使用与 CPU 资格完全相同的 installed wheel，
SHA-256 为 `8068a1d916a8c7939068b53bb496fe768e7e150f58a8c2d3d367b85ecf77849f`。

## 唯一 H100 Job

- Job ID：10902665
- State / ExitCode：COMPLETED / 0:0
- Elapsed：00:00:27
- Node：gpu-h100-4-0
- GPU：NVIDIA H100 PCIe
- Requeue / Restarts：0 / 0
- formal_h100_submission_count：1
- automatic_retry：false

环境门禁确认：PyTorch 2.5.1+cu121、CUDA runtime 12.1、实际分配设备 cuda:0、TF32 关闭、无源码 shadow import、pip check PASS、checkpoint workspace 位于 NFS `0x6969`。

## 科学状态与 restart 门禁

Periodic 和 Channel functional runtime 均完成 4 步连续路径以及 `2 + checkpoint + restart + 2` 路径。两种 family 的最终 state、Q、velocity 和 pressure 全部逐字节一致，全部数组有限，checkpoint import 精确，重复目标被拒绝且未修改既有目标，provider identity 前后不变，临时残留为 0。

## 归档

控制目录：

`/home/fansenwei/pssolver_v021_n1_h100_15670eb_20261020_v1`

scratch：

`/scratch1/vincent/PSSolver/data_pssolver_v021_n1_h100_15670eb_20261020_v1`

- 控制目录：25/25 checksum PASS；manifest SHA-256 `694043a8a8f01dba5506482d32273d1cc65ef2ad9fc4290a446d6002f1b26e0c`
- scratch：18/18 checksum PASS；manifest SHA-256 `2ec118f3c089a25377cbccad3da8d363bbb30d0cee61f42049d250f1076e6742`
- `COMPLETE` 已写入。

## 最终边界

`qualification_complete = true`，`n1_complete = true`，`eligible_for_v0_2_1_finalization = true`。

没有修改求解器方程、数值算法、checkpoint schema、Functional API、provider identity、生产默认值、PSSolver-Control 或 nematics3d。版本号仍为 0.2.0，这是预发布资格候选的预期状态。下一步是 push 本证据记录，然后进行仅包含版本号、发布说明与资格 successor 的 v0.2.1 发布准备；tag、GitHub Release 和 PyPI 均仍需另行授权。
