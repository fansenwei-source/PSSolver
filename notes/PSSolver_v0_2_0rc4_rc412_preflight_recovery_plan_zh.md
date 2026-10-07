# PSSolver RC4.1.2 H100 preflight 可诊断恢复计划

## 1. 原因

Job `10859008` 已经证明 scheduler 配置有效，但外部 helper 把 P01–P08 写成同一个 `python -c` 裸断言序列，并把 observed values 安排在全部断言通过之后才持久化。首个 false predicate 因而只留下无 ID 的 `AssertionError`。后续只读分析正确地将结果判为 `UNRESOLVED_REQUIRES_DECOMPOSED_H100_PREFLIGHT`。

这不是 solver 或 scientific failure，也没有运行 CUDA tests、storage diagnostic、profiles 或 analyzer。B5 仍未关闭。

## 2. 版本化 helper

新增 `benchmarks/capture_rc4_h100_preflight.py`。它不修改求解器运行时代码，而是把 preflight 变为以下确定流程：

1. 创建严格 JSON 报告，记录 cwd、Python、PyTorch、CUDA runtime、关键环境变量和 import 路径；
2. 原子记录两个 TF32 runtime flag 的原始值；
3. 显式执行 `torch.backends.cuda.matmul.allow_tf32 = False` 和 `torch.backends.cudnn.allow_tf32 = False`；
4. 原子记录设置后的两个 flag；
5. 独立求值 P01–P08；每完成一项就通过同目录临时文件、flush、fsync、`os.replace` 和 JSON 回读更新报告；
6. 即使某项失败，也继续收集其余 predicate；
7. 只有完整报告持久化后才 enforcement，并在异常中包含 predicate ID、observed、expected 和 exception。

helper 拒绝覆盖已有输出，不使用裸 `assert`，也不把 `NVIDIA_TF32_OVERRIDE=0` 当成两个 Python runtime flag 的替代证据。

本地扩展定向门禁为 `76 passed`。完整 CPU suite 为 `2790 passed, 7 deselected, 8 subtests passed`，0 failure、0 skip、0 xfail，运行时间 203.69 秒。

## 3. P01–P08

- P01：`pssolver` 来自精确 Q worktree；
- P02：Plane Nyquist diagnostic 来自精确 Q worktree；
- P03：PyTorch CUDA 可用；
- P04：只看到一张 CUDA GPU；
- P05：实际分配身份为 `cuda:0`；
- P06：GPU 名称包含 `H100`；
- P07：显式设置后 matmul TF32 为 false；
- P08：显式设置后 cuDNN TF32 为 false。

## 4. 本阶段边界

本地阶段只实现、测试、记录并 push helper，不提交 Slurm Job。此前一次正式 H100 Job 仍计为 1；新的 H100 recovery 必须在 Q2 身份和 CPU 门禁冻结后由用户另行明确授权。

Q2 不修改 `pssolver/`、PSSolver-Control、`nematics3d`、科学阈值、profile 合同或生产默认值。它也不授予 B5 PASS、RC4.2 规划资格、默认提升或发布资格。
