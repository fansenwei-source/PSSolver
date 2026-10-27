# PSSolver v0.3.0 S-10 最后清理候选

本候选从 `f3620b8` 独立开分支，不修改仍在排队的 S-09 分支，也不触碰 HPCC。

`V-33` 的数值构建指纹升级为 schema 2。Torch 指纹只删除四条会随 CUDA
可见性出现或消失的探测摘要，仍保留包含 CUDA、编译器、BLAS、MKL 与 CPU 能力的
build-settings 行；NumPy 改为哈希结构化配置。v0.2.0 与 S-08 的 checkpoint、snapshot
和 continuation 数组均未改变。

`R1-R9` 按 `D-0011 / Q-06 A` 只处理可独立验证的子集：R7 统一三个 public compiler
的 exact-key 机制，同时由调用者继续拥有异常类型和 rejection code；R9 把 `api`、
`boundaries`、`diagnostics`、`functional` 纳入可执行依赖棘轮。其余项目涉及公共构造、
发布 checkpoint、独立 Consumer、数值 hot path、弃用/lifting 或证据保留政策，保持延期，
没有为了“清零”而混入本轮。

`V-33` 当前为 `claimed_fixed`，`R1-R9` 仍为带逐项理由的 `deferred`。本地完整 CPU
证据完成后以 `R-0032` 交 Claude 独立复核；复核前不推送。
