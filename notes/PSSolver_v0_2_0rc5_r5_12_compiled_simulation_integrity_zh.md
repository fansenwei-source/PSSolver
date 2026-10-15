# PSSolver v0.2.0rc5 R5-12：CompiledSimulation 一致性封印

## 当前状态

`classification = PASS_R5_12_COMPILED_SIMULATION_INTEGRITY_CPU_CLOSURE`。

实现、完整源码 CPU 回归和 fresh installed-wheel CPU closure 均已完成。qualified source commit 为 `3d62328e1b5b6962e9a2325d1a6b0165940b3665`；`eb8aed1c00ae624a0172667473aae1f189c83e5c` 是核心实现提交，后续提交只维护冻结历史证据的前向 successor 链。

## 修复

`compile_simulation()` 现在为编译产物生成一个稳定 SHA-256，绑定源 Simulation、应用身份、降级后的 application specification、lowering plan、construction plan、application request 的类型与内容，以及 normalization。`run_simulation()` 在选择 application runner 之前验证该身份；`to_metadata()` 在输出 provenance 之前也执行同一验证。

因此，用 `dataclasses.replace()` 替换 `application_request` 或其他任一已绑定字段时，旧封印不会随之改变，执行会以 `ValueError: compiled simulation integrity mismatch` 失败。拒绝发生在 runtime 构造、科学 timestep 和输出目录创建之前。

该设计不会在运行时重新编译，也不会在 timestep 热路径增加检查；每次 `run_simulation()` 只在进入 application runner 前验证一次。

## 验证

R5-12 定向 CPU 门禁为 `41 passed`；覆盖冻结记录解析器的集中门禁为 `275 passed`。最终完整源码 suite 为 `3599 passed, 7 skipped, 8 subtests passed, 0 failed`。

从 qualified source 的 clean `git archive` 构建 wheel，并安装到新的隔离 venv。`pssolver` 从该 venv 的 `site-packages` 导入，source shadow import 为 false，`pip check` 通过。installed-wheel 定向门禁为 `46 passed`；适用完整 suite 为 `3592 passed, 7 skipped, 8 subtests passed, 0 failed`。唯一排除的是既有的 source-layout 文件 `tests/test_phase8_p861_local_cumulative_audit.py`，其中 7 个测试没有被伪装成 skip 或 deselect。

installed full suite 的第一次调用误把控制目录设为 pytest 工作目录，导致 7 个需要仓库相对路径的架构审计找不到文件。它没有被归类为实现失败。按既有 runner 合同改为“先导入 wheel、再追加仓库测试支持路径、pytest cwd 使用仓库根”后，上述 7 项与全套测试均通过。

证据目录为 `/home/fansenwei/pssolver_rc5_r512_installed_cpu_3d62328_20261015_v1`；checksum 为 `9/9 PASS`，`checksums.sha256` 自身 SHA-256 为 `5fffbe3411b1dd17d024bafbdc7c0c973932a406be4814ffaedfd9798724e69f`。

## 范围

本阶段不修改科学方程、数值热路径、checkpoint schema、生产默认值、PSSolver-Control 或 `nematics3d`；不需要 H100，也没有 push。
