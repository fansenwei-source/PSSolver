# PSSolver RC4.4.7 单次 H100 Channel pressure 累计关闭计划

RC4.4.7 是 RC4.4 的最终资格阶段，而不是 Consumer 分支的合并步骤。本计划固定 A 栈为 PSSolver `5071d73` 与 PSSolver-Control `03dbc39`，B 栈为 Provider runtime `dc8c87a` 与 Consumer source `0cf55b4`；后续文档提交只登记证据，不替代这些已资格 source identity。

正式作业必须从四个固定 wheel 建立隔离环境，并只通过 installed packages 运行。Candidate 同时覆盖 fixed-work 与 tolerance 两种 pressure policy、三个网格与指数 `-600/0/600` 的 primal/transpose 尺度门禁、custom adjoint、有限差分、resolution-aware Taylor、checkpointed/full-history gradient、生产与 functional same-version restart，以及旧 identity 和篡改 policy 的 pre-mutation rejection。

性能矩阵固定为 `R128×20×20` 和 `R512×40×40`，A/B 各三次，顺序为 A/B、B/A、A/B。每项运行 10 个 warmup 与 50 个正式步骤，要求 steady-state transform calls 为 11/36、fallback 和 graph break 为零；mean/median timestep ratio 不超过 1.05，peak allocated ratio 不超过 1.05，reserved 只作 allocator 诊断。只有 pressure iteration 与 termination 路径匹配的配对才要求 final-state SHA 完全一致。

qualification-only 执行支持现已实现，包括 7 项 CUDA-only test runner、候选 CUDA correctness runner、12 项 production profile、fail-closed 累计 analyzer 和纯 command planner。所有 JSON 都采用原子写入、禁止覆盖和严格 NaN/Inf 拒绝；runner 将 CPU-only unrolled oracle 与 H100 科学证据明确分层。合成测试为 15/15，通过同轮相关 pressure 测试 95/95。

固定本地环境中的最终完整 CPU suite 为 `3165 passed, 7 deselected, 8 subtests passed`，0 failure、0 skip、0 xfail，用时 204.77 秒。七个 deselect 精确对应既有 CUDA-only node ID；这些节点被版本化 CUDA runner 固定，并必须在未来唯一 H100 Job 中得到 7/7 PASS。

为防止 installed-wheel 资格验证被源码遮蔽，正式执行必须把固定的 `benchmarks/` 支持文件复制到一个不含 `pssolver/` 包的外部 support root，然后分别使用 A/B 新 venv 的 Python 启动。command planner 固定 12 个唯一 profile、A/B–B/A–A/B 顺序和每 profile 独立 Triton/TorchInductor cache；它本身不执行命令。

当前仍未实现或授权 Slurm submission，也未运行任何 H100 资格门禁。下一步是本地提交本组 qualification-only 支持；push 与唯一一次 H100 submission 都需要用户另行明确授权。不得自动重试，也不得根据正式结果修改阈值。

该阶段不修改 Provider runtime、Consumer runtime、生产默认值或 `nematics3d`，不授权 larger batch、optimizer campaign、科学 control 结论、自动 merge、默认提升或发布。
