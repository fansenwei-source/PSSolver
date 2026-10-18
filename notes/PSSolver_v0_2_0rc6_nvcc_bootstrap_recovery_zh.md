# PSSolver RC6 nvcc bootstrap recovery

Job `10900727` 的 node-local `/tmp`、`RENAME_NOREPLACE` 和 installed-wheel
preflight 已通过，但第一个 environment stage 因 batch `PATH` 中没有 `nvcc` 而停止。
该事件归类为 qualification harness 的 CUDA-toolkit bootstrap 错误；没有启动 CUDA
tests、科学 timestep、checkpoint、profile 或 Consumer gate。

本 recovery 只修改资格支持层：Slurm 脚本显式加载 HPCC 的 `cuda/12.1` module，
并仅在加载站点 Lmod 初始化脚本期间临时关闭 Bash `nounset`，加载后立即恢复，
并在任何资格 stage 前运行版本化 preflight。preflight 原子记录 module、`nvcc`
路径与版本、CUDA header、`CUDA_PATH`、PyTorch CUDA runtime 和 Slurm 身份；即使判定
失败也会先写出 observed JSON，再以非零状态停止。

旧 `PSSolver_v0_2_0rc6_qualification_support.json` 保持未改写。Slurm 脚本的新哈希
通过 forward-only successor 记录登记。没有修改 PSSolver runtime、公共 API、数值实现、
阈值、checkpoint schema、PSSolver-Control、nematics3d 或生产默认值。

本地验证完成后只允许准备新的 recovery_v2 support/control/scratch 和冻结 plan；push 与
新的 H100 recovery submission 都需要单独明确授权，不得自动重试。
