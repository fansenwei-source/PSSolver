# PSSolver v0.4 S-18 R2 checkpoint/integrity 候选

本地实现提交为 `f83539e76cf027c37010205f03b7b4e4f599002e`，父提交为
`eb6a191f83ec7bc59ed41dde0b6d67dab57acc10`。本批按 D-0018 与
R-0054 已验收的设计，把重复的纯 artifact mechanics 收到私有模块
`pssolver.io._checkpoint_artifacts`。

共享范围包括历史 indented canonical JSON、原始文件 SHA-256、标准
NumPy tensor record、provisional reference tensor record 及 finite-Q 的
size-bearing record。各 checkpoint family 原有的 wrapper、operation、异常翻译、
schema、reader 和 migration 保持不变。

本批没有修改 target 创建、staging 位置、placeholder 所有权、fsync、
rename/renameat2、清理、符号链接/设备判断、`checkpoint_directory_transaction`
实现或发布调用图，因此没有触发真实 NFS 复测条件。

验收必须同时证明：相对 v0.3.0 与本批父提交的当前 writer 目录树逐字节相同；
全部已发布 checkpoint 只读兼容；8 family × 20 步状态与身份逐位相同；两套完整
CPU、屏蔽外部路径、故障矩阵、clean-wheel 与 Control main installed-wheel
门禁全部通过。没有授权非零容差，也没有授权 push、发布、HPCC、Slurm、GPU
或 H100。
