# PSSolver RC4.4.7 单次 H100 Channel pressure 累计关闭计划

RC4.4.7 是 RC4.4 的最终资格阶段，而不是 Consumer 分支的合并步骤。本计划固定 A 栈为 PSSolver `5071d73` 与 PSSolver-Control `03dbc39`，B 栈为 Provider runtime `dc8c87a` 与 Consumer source `0cf55b4`；后续文档提交只登记证据，不替代这些已资格 source identity。

正式作业必须从四个固定 wheel 建立隔离环境，并只通过 installed packages 运行。Candidate 同时覆盖 fixed-work 与 tolerance 两种 pressure policy、三个网格与指数 `-600/0/600` 的 primal/transpose 尺度门禁、custom adjoint、有限差分、resolution-aware Taylor、checkpointed/full-history gradient、生产与 functional same-version restart，以及旧 identity 和篡改 policy 的 pre-mutation rejection。

性能矩阵固定为 `R128×20×20` 和 `R512×40×40`，A/B 各三次，顺序为 A/B、B/A、A/B。每项运行 10 个 warmup 与 50 个正式步骤，要求 steady-state transform calls 为 11/36、fallback 和 graph break 为零；mean/median timestep ratio 不超过 1.05，peak allocated ratio 不超过 1.05，reserved 只作 allocator 诊断。只有 pressure iteration 与 termination 路径匹配的配对才要求 final-state SHA 完全一致。

本步骤只冻结计划。qualification runner、analyzer、Slurm script 和正式归档尚未实现；H100 submission 也尚未获得授权。后续应先实现并通过 CPU synthetic/preflight，再单独请求一次 H100 授权。不得自动重试，也不得根据正式结果修改阈值。

该阶段不修改 Provider runtime、Consumer runtime、生产默认值或 `nematics3d`，不授权 larger batch、optimizer campaign、科学 control 结论、自动 merge、默认提升或发布。
