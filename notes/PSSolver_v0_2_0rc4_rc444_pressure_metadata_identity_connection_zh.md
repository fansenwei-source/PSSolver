# PSSolver v0.2.0rc4 RC4.4.4：Channel pressure metadata 与 identity 连接

RC4.4.4 已把 RC4.4.2 冻结、RC4.4.3 实现的 pressure policy 接到 Provider 的公共声明、公共编译结果、生产/functional diagnostics、capability metadata 和 checkpoint identity。

用户输入仍然只有 `algorithm`、`relative_tolerance`、`max_iterations`、`fixed_iterations`、`warm_start` 五项。`convergence_mode`、authoritative iteration limit、active-mode RMS、二次幂 working scaling、relative breakdown 和 initial-guess policy 都是只读派生 metadata，不能作为额外输入绕过编译合同。

生产 forward identity 和 functional forward/derivative identity 已分别升级为 v4；state layout、backend restart、functional bridge envelope 保持原版本。旧 v3 forward/derivative identity 会在打开 tensor 和修改目标前被拒绝；本阶段没有添加静默迁移。

声明与 runtime capability set 现在完全一致。`explicit_vjp=false` 仍表示没有独立的公共 explicit-VJP 方法；实际可微性来自 executable runtime 内的 custom implicit pressure adjoint。component metadata 已移除含义模糊的 `functional_runtime_executable` 字段。

本阶段没有改动 PSSolver-Control、nematics3d、生产默认值，也没有 push 或提交 H100 作业。installed-wheel CPU、Consumer migration 和单次 H100 累计关闭仍属于后续阶段。
