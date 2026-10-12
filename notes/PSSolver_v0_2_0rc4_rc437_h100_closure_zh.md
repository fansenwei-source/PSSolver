# PSSolver RC4.3.7 internal lifting H100 关闭记录

RC4.3 已完成，B8、B20 和 N4 从 `partial_residual` 转为 `closed_verified`。权威分类为：

`PASS_RC4_3_7_INTERNAL_LIFTING_SINGLE_H100_CLOSURE`

## 关闭方式

RC4.3.1 证明现有 homogeneous-Dirichlet DST remainder 不能支持公开的非零壁面 molecular-field 收敛声明，且历史 whole-slab sine taper 会无条件改变整个初态。RC4.3.2–RC4.3.5 随后采用 fail-closed 路线：公开 compiler 在构造 runtime 或输出之前拒绝未经资格化的 lifted Beris–Edwards 组合；fresh conditioning 改为显式策略；fresh-only provenance 与同版本 checkpoint identity 分离。

因此，本关闭记录不声称原来的非收敛数学路径已经被修复，也不重新开放 public lifted Plane。它确认限制合同、field-neutral lifting infrastructure、显式 localized fresh conditioning、provenance 和同版本 restart 均已实现并通过相应资格门禁。

## CPU 与 installed-wheel 证据

RC4.3.6 已在 source checkout 和 fresh installed-wheel 环境中关闭 CPU 门禁。最终 recovery support 的本地定向测试为 157 passed；完整 suite 为 3045 passed、7 个精确 CUDA-only deselect、8 subtests passed，零 failure、skip 和 xfail。HPCC qualification-support 测试为 20 passed。

## 唯一 H100 作业

唯一作业 `10875343` 在 `gpu-h100-4-0` 上以 `COMPLETED / 0:0` 结束，用时 `00:03:48`。本轮只提交一次正式 H100 Job，没有自动重试或第二个 Job。环境为 NVIDIA H100 PCIe、PyTorch 2.5.1+cu121、CUDA runtime 12.1；TF32 matmul 与 cuDNN 均关闭。

P01–P08 preflight 与 7 个 CUDA-only tests 全部通过。correct-layer 门禁位于 `direct_conditioning_operator_output_before_spectral_projection`：

- interior 与未调理输入逐字节一致；
- transition region 确实改变；
- CPU/CUDA relative L2 为 0；
- wall residual Linf 为 0；
- 4-step continuous 与 2+restore+2 同版本 restart 逐字节一致；
- 全部状态有限。

post-projection 比较只保留为诊断，不再错误地作为 localized conditioning 的空间局部性判据。

## 性能与显存

12/12 profile 全部完成 60 steps，forward/inverse transforms 每步为 7/32，graph breaks 为 0，runtime/compile fallback 均为 false。

- R128：mean ratio `0.9938881735794637`，median ratio `0.995631180817086`；
- R320：mean ratio `1.001859660283416`，median ratio `1.0023452469046086`；
- 两个网格的 peak allocated 与 peak active ratio 均为 `1.0`；
- peak reserved ratio 仅作诊断，两个网格均为 `1.0`。

全部结果通过冻结的时间门限 1.15 和 allocated/active 门限 1.10；配对初态与最终状态身份一致。

## 归档

控制目录：

`/home/fansenwei/pssolver_rc437_lifting_h100_4d176b2_20261011_recovery_v3`

Scratch：

`/scratch1/vincent/PSSolver/data_pssolver_rc437_lifting_h100_4d176b2_20261011_recovery_v3`

统一 manifest 为 1105/1105 PASS，SHA-256：

`204c1b46cd9b4212a75743d17df1efb290a3cd6c805cb943df464665471cea65`

`COMPLETE` 已生成。A、B、Q3 worktree 均保持 detached、精确 HEAD、clean 且无缓存。失败的 recovery v2 归档保持未修改，206/206 checksum PASS。

## 下一步边界

只允许进入 RC4.4 pressure contract 的规划阶段。RC4.4 将处理 B25、B28 和 B29，并需要在 Provider 合同冻结后协调 PSSolver-Control 的梯度、Taylor、checkpoint 与 H100 重新资格。当前不授权 RC4.4 实现、Control 修改、自动 merge、默认值提升或发布；`nematics3d` 未修改。
