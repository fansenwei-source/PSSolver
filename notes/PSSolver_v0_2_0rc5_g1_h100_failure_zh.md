# PSSolver v0.2.0rc5 G-1.9：首次 single-H100 资格结果

最终分类为 `FAIL_RC5_G1_H100_CUDA_ONLY_CHECKPOINT_FILESYSTEM_PORTABILITY`。G-1 尚未完成，rc5 尚不可发布。

唯一正式作业 `10885517` 在 `gpu-h100-4-0` 上运行 41 秒后失败。本轮只调用一次 `sbatch`，没有自动重试或第二个 Job。

Environment 阶段完整通过：H100、`cuda:0`、三套 installed-wheel venv、wheel 身份、`pip check`、TF32-off 和 source-free import 均符合冻结合同。CUDA-only 阶段的七个测试有六个通过，一个失败，没有 skip、xfail 或 collection error。

失败测试是 finite-Q anchoring 的 CUDA continuous/split/file-restart 测试。CUDA timestep 已完成；失败发生在 checkpoint 目录发布阶段。`pssolver.io.checkpoint._rename_directory_no_replace` 直接要求 `ctypes.CDLL(None).renameat2`，而 HPCC 使用 Linux 3.10 和 glibc 2.17，没有该动态符号，于是抛出 `CheckpointIOError: atomic no-replace checkpoint publication is unavailable`。

这是 Provider checkpoint 文件系统兼容性缺口，不是外部 qualification helper、scheduler、CUDA、TF32、数值有限性或科学模型失败。后续八个阶段均未运行，`COMPLETE` 未写入。

失败归档位于 `/home/fansenwei/pssolver_rc5_g1_h100_674a812_20261015_v1`，`20/20` checksum 通过，manifest SHA-256 为 `f2b1ea3e68be47c29ca2fb92d691b8d298c9209daf466bbba2b902ed9acbb2a4`。

下一步应实现一个兼容 glibc 2.17、同时保持 transactional 和 no-overwrite 合同的 checkpoint 发布路径，并完成 CPU 回归。任何新的 H100 recovery 都需要新的明确授权。
