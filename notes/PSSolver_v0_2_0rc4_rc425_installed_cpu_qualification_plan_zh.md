# PSSolver RC4.2.5 installed-wheel CPU 资格计划

RC4.2.5 不再修改 checkpoint reader、writer 或 runtime。它从精确 Git commit 构建 PSSolver 与 PSSolver-Control wheel，在一个新的 CPU venv 中用 `--no-deps` 安装，然后从 source checkout 之外运行冻结的 95 项门禁。

Provider 的 35 项测试覆盖 RC4.2.0 冻结的 M01–M16、当前格式 round-trip、注册旧格式接受、不兼容旧格式拒绝、升级工具 source immutability 和 pre-mutation rejection。Consumer 的 60 项测试覆盖 Periodic/Channel public adapter、functional checkpoint round-trip、bitwise replay、checkpointed/full-history gradient、有限差分、Taylor 与错误 identity 拒绝。

installed runner 必须正面证明 `pssolver` 与 `pssolver_control` 都来自新 venv 的 `site-packages`，且 `sys.path` 不含两个 source root。95 项必须全部通过，不允许 skip、xfail、collection error 或 source shadow。结果只能原子写入一个此前不存在的 JSON。

本阶段是 CPU-only，不提交 H100，不修改 PSSolver-Control、`nematics3d`、生产默认值或 Git 远端。installed-wheel gate 通过后仍需运行 Provider 完整 CPU suite；RC4.2.6 是否执行由后续是否声称 CPU↔CUDA checkpoint portability 决定。
