# PSSolver v0.2.0rc5 G-1 HPCC 预提交准备

本轮已经完成 HPCC 的无 GPU 预提交底座，但尚未冻结九阶段计划，也没有提交 H100 Job。

在新的 scratch 根目录中，rc4 baseline、rc5 candidate 和 PSSolver-Control 三份冻结 wheel 经过远端 SHA-256 二次验证后，分别安装到三个 fresh `--system-site-packages` venv。三个环境均使用 Python 3.10.18，包均从各自 venv 的 `site-packages` 导入，三次 `pip check` 均通过。Provider qualification support 中不包含 `pssolver/` 源码树；预定的正式 H100 输出目录仍不存在。

预检同时发现当前 support commit `3443f923b942e586a221241967823162a302d8ab` 只有 TF32、profile、assemble 和 final analyzer 的实际 producer。`environment`、`cuda_only`、`checkpoints`、`numerical` 和 `consumer` 五个阶段目前只有 analyzer 所要求的 schema，没有能生成这些 schema 的版本化 executable producer。此时用占位命令冻结计划会得到一个已知不可运行的 H100 作业，因此流程按 fail-closed 原则停止在提交前。

下一步是只在 `benchmarks/**`、`tests/**` 和 forward-only `notes/**` 范围内补齐并进行 CPU/synthetic 验证。不得修改 `pssolver/**` 数值实现、PSSolver-Control、nematics3d、生产默认值或科学阈值。补齐后应重新生成 support archive，在 HPCC 重新执行 SHA 绑定和 `--preflight-only`；push 和唯一一次 H100 submission 仍需单独授权。
