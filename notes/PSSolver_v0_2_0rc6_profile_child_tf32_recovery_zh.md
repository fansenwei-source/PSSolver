# PSSolver v0.2.0rc6 profile child TF32 bootstrap recovery

## 结论

本修正只处理 RC6 recovery v6 的外部资格编排缺口：192 个 profile 子进程在进入既有 profiler 前，没有显式关闭 Python runtime 的 cuDNN TF32 flag。它不修改求解器、数值算法、既有 profiler、门限、公共 API、checkpoint schema、PSSolver-Control、nematics3d 或生产默认值。

当前分类为 `READY_RC6_RECOVERY_V7_PROFILE_CHILD_TF32_BOOTSTRAP_H100_NOT_AUTHORIZED`。这不是 RC6 qualification PASS，也不授权 push 或 H100 submission。

## v6 失败证据

Job 10901220 完成了 11 个正式阶段中的前 10 个阶段。192/192 profiles 均生成且 finite；Consumer 为 344 passed、1 个既有 benchmark skip、0 failed。最终 analyzer 在第一份 profile 上拒绝了 `cudnn_allow_tf32=true`。控制归档 23/23 checksum PASS，正式归档 4567/4567 checksum PASS；旧分类保持不变。

## 修正合同

新增的 profile-child wrapper 在构建任何 runtime 前：

1. 读取并记录三个 Python TF32 状态；
2. 显式设置 matmul precision 为 `highest`、CUDA matmul TF32 为 false、cuDNN TF32 为 false；
3. 再次读取状态，并把设置前后值、逐项异常、profile 脚本路径和 SHA-256 原子写入独立 JSON；
4. 只有 JSON 已持久化、回读一致且三项状态精确满足合同时，才调用既有 profiler。

Matrix runner 为 192 个 profile 生成 192 个互不重复的 bootstrap 路径，逐项验证 schema、状态、异常列表、脚本路径及哈希，并把 profile SHA-256 与 bootstrap SHA-256 一起写入进度记录。最终 `MATRIX_COMPLETE.json` 必须同时确认 192 份 profile 和 192 份 bootstrap 证据。

## 边界

既有 `profile_rc458_runtime_family.py` 保持不变，因此计时、显存、transform counter、runtime 构建和 analyzer 门限均未变化。本轮先完成本地测试和 HPCC 提交前冻结，不 push、不提交 H100。后续必须由用户另行明确授权一次 recovery v7 H100 submission，且不得自动重试。
