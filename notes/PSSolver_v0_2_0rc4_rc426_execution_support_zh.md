# PSSolver RC4.2.6 跨设备 checkpoint 执行支持

状态：`READY_RC4_2_6_SINGLE_H100_EXECUTION_NOT_SUBMITTED`

本步把 RC4.2.6 的冻结计划实现为可执行、可审计的资格工具，但没有提交 H100 Job。

## 已实现内容

- 冻结的 X01–X14 矩阵执行器，顺序固定为 7 组 runtime 的 CPU→CUDA 与 CUDA→CPU。
- 每个 cell 单独原子写入 JSON；任一 cell 失败后立即停止，未通过全部 14 项时不写 `MATRIX_COMPLETE.json`。
- 每项验证 checkpoint compatibility identity、文件校验和、source tree 不变、进度计数、序列化 tensor payload、persistent backend payload、设备身份、有限性与 `fallback_used=false`。
- 每项均制作 raw-byte tamper 副本，要求到达 checksum/integrity guard，且拒绝前 target 保持未修改。
- 恢复后仅要求再执行一步且保持有限；不声称 CPU/CUDA 后续轨迹逐字节一致。
- 独立 fail-closed aggregate analyzer。
- 严格统计 7 个 CUDA-only node 的 runner，不允许 skip、xfail、xpass、deselect 或 collection error。
- 单 H100 Slurm 模板，固定 `hagan-lab / hagan-gpu / medium / gpu:H100:1`，显式关闭 TF32，使用独立 Triton/TorchInductor cache，并禁止脚本内部再次提交作业。

## 本地验证

- 定向测试：28 passed。
- 完整 CPU suite：2912 passed，7 deselected，8 subtests passed，0 failed，0 skipped，0 xfailed，用时 198.08 秒。
- 7 种 runtime 族均在 CPU 同设备模式下真实完成写入、tamper 拒绝、恢复、payload 精确比较和恢复后一步 smoke。
- `git diff --check`：PASS。

## 边界

本步没有修改 runtime、checkpoint reader/writer、PSSolver-Control、nematics3d 或生产默认值；没有运行 H100、提交 Slurm、push、merge 或发布。

下一步应先在 HPCC 登录节点建立 fresh installed-wheel venv，将经过 SHA-256 验证的 qualification-only `benchmarks` 和 7 个测试复制到该 venv 内，完成 Git/RC4.2.5/CPU/wheel/helper/path/empty-scratch 门禁。通过后，再单独授权唯一一次 RC4.2.6 H100 submission。
