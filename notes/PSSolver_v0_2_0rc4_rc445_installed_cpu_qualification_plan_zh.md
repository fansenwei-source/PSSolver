# PSSolver RC4.4.5 Provider installed-wheel pressure CPU 资格计划

RC4.4.5 从精确 Git commit 的 clean archive 构建 wheel，并在源码目录之外的新 venv 中以 `--no-deps` 安装。runner 必须正面证明 `pssolver` 来自该 venv 的 purelib、`sys.path` 不含源码 checkout、`pip check` 通过，并将结果原子写入 JSON。

source 与 installed-wheel 使用同一组 85 项行为测试，覆盖 B25 的极端尺度、primal/transpose、breakdown、dense/unrolled gradient、有限差分和 Taylor；B28 的 declaration/runtime capability 与 diagnostics schema v2；以及 B29 的 fixed-work/tolerance 语义、warm/zero start、v4 identity、同版本 checkpoint 和旧或篡改 identity 的 pre-mutation 拒绝。

本阶段只验证 Provider。它不修改数值热路径、checkpoint 接受规则、PSSolver-Control、`nematics3d` 或生产默认值，也不运行 H100。通过后才能进入独立的 RC4.4.6 Consumer migration；RC4.4.7 单次 H100 累计关闭仍需要用户另行明确授权。
