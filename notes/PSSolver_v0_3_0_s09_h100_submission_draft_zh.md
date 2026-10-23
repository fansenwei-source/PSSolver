# PSSolver v0.3 S-09 single-H100 提交授权草稿（待 R-0022 复核）

本文件不是授权，也不表示已经提交 H100。只有 R-0022 返回 PASS 后，才能依据 D-0009 把下列内容转成给用户的一键授权文本。

## 绑定身份

- Provider 运行时资格源：`dc12dc6fbf984fb1a98e1de672903e51750742f3`
- Provider qualification-support 源：以 R-0022 验收的最终本地分支头为准；不得改变上述运行时资格源
- Consumer 稳定基线：`208edd03e6b26a0e788c07d295fddd7a67237189`
- Consumer migration：`48dd4d3a9a101f8b24d8311c57ed67fab46ceea0`
- Consumer evidence：`97d3c06450c4a7a6d72e0fe07a7a157ef095c133`
- v0.2.2 baseline wheel：`a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e`
- Provider candidate wheel：`b3cb41689e5ac180b7dc8b900d84e2357b731976d40e01bedfd742e8688415b1`
- Consumer candidate wheel：`40ba784811ca1e9f3da2a2a2768b44d752a5972aa40217a9ef36f30dcbea1ace`
- Provider source archive：`876fcf4b6234892fb2d6f6f3fe895d83b74d8cb0b4a9a752760aa34efe01b2d2`
- Consumer source archive：`02f3e8249bc02cbb0484636d9f8bf46cc8a6815d90e13f8286f7aaf8291157fd`
- Consumer support archive：`2243f14cbdad1a58cb8a1ff73e2bb550752d97f5edf2190302005d9d69ff66a1`
- 冻结 H100 plan SHA-256：`80e8c7f159e6cef1ea7a82937844e0a9795fdd880c8b684ea46c9a7ad1aee7d7`
- Slurm script SHA-256：`08b354d2c897741755fbf5629960a61d20f2002dab769336afad3941b3ed389c`

## 拟授权范围

授权上传上述只读 archive、wheel、资格 helper 和 job script 到 HPCC，并提交一次 PSSolver v0.3 S-09 single-H100 cumulative qualification Job。不得自动重试，不得提交第二个 Job。

作业必须使用：`hagan-gpu`、`hagan-lab`、`medium`、`gpu:H100:1`、8 CPUs、64G、3 小时、`--no-requeue`。在任何资格阶段前完成 node-local `/tmp`、`RENAME_NOREPLACE`、host compiler、installed-wheel import、CUDA device identity 和 TF32 初始状态门禁。任何门禁失败必须立即停止并完整归档。

阶段顺序固定为：环境/编译器 → 7 项 CUDA-only tests → 三个 CUDA t=200 Hermitian 长时程 → forward/backward TF32 scope → checkpoint/restart/released fixture → 数值/梯度/Taylor/replay → 独立 Control consumer → 192 项 performance/memory profiles → cumulative analyzer/checksum。独立 Control consumer 必须先于性能矩阵。

性能合同保持冻结：8 runtime families × 2 grids × 6 trials × baseline/candidate = 192 profiles；每个 profile 60 个正式样本；交替 A/B 顺序；全部样本保留；稳定分块中位比必须 `<= 1.03`，每个 paired ratio 必须 `<= 1.05`；不得删样本、替换 trial、放宽门限或在失败后自动重跑。

作业必须持久化完整环境、输入/源码/运行时文件/wheel/RECORD 哈希、全部 JSON、pytest 日志与 JUnit XML、Slurm provenance、checksums.sha256；所有门禁通过并复核 checksum 后才能写 COMPLETE。A3-18 与 NEW-14 只有在该 H100 结果通过且再经 Claude 独立验收后才能改为 verified。

本授权不包含 push、tag、Release、PyPI、PSSolver-Control push、生产默认值修改、阈值修改或科学 Control 结果声明。
