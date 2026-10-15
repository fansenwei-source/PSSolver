# PSSolver v0.2.0rc5 G-1.7 九阶段 producer 补全记录

日期：2026-10-15

分类：`READY_RC5_G1_NINE_STAGE_PRODUCER_SET_FOR_HPCC_SUPPORT_REGENERATION`

本批次补齐了 G-1 九阶段计划中此前缺失的五个版本化 producer：environment、CUDA-only、checkpoint、numerical 和 Consumer。已有的 TF32、profile、assemble 与 analyzer 保持不变。

所有新 producer 都能以绝对脚本路径直接启动，自行建立只读 support-root import bootstrap；它们拒绝覆盖既有产物，并使用原子 JSON 写入。A/B numerical 由各自 installed-wheel Python 子进程独立生成状态，父进程不导入任一 Provider wheel。

Checkpoint producer 覆盖 Periodic functional、Channel functional fixed-iteration、Channel functional tolerance 与 legacy Channel，正式计划为 CPU→CUDA 和 CUDA→CPU 两个方向。登录节点 CPU→CPU 真实冒烟的四个 family 全部通过，包括 payload tamper、identity mismatch、拒绝前 target 不变、源 checkpoint 不变和精确 replay。rc1–rc4 的 13 项 released disposition 继续绑定既有只读 CPU record。

Numerical producer 覆盖八个 runtime family。本机 CUDA 两步真实冒烟为 8/8；两个 repaired Periodic family 的 candidate Hermitian violation 都是精确 0.0。六个未改 runtime 使用 byte identity；两个 Periodic repair 使用 physical Q、velocity、pressure 的相对 L2/Linf 门禁。

本批次相关合成与既有定向测试为 56 passed；foundation producer 文件为 27 passed；py_compile、直接脚本 `--help` bootstrap 和 `git diff --check` 均通过。

本批次没有 push、没有 H100 submission、没有改动生产 runtime、科学阈值、默认值、PSSolver-Control 或 nematics3d。

下一步：从本提交重新生成 source-free HPCC support archive，在已安装的三个 wheel 环境上冻结并静态验证真实九阶段 argv 计划；仍不提交 H100 Job。
