# PSSolver RC4.3.7 profile JSON serialization recovery 计划

Job `10874753` 已经证明 correct-layer CUDA/restart 门禁通过，但第一个 baseline R128 profile 在完成 60 步并写出有效 JSON 后，被 profiler 自身的 Python 对象严格比较错误拒绝。`RuntimeProfileConfig.shape` 和 `lengths` 在 `dataclasses.asdict()` 后仍是 tuple，而 JSON 回读必然得到 list；这不是科学、数值、CUDA、性能或显存失败。

本 recovery 只修改 `benchmarks/profile_rc437_plane_runtime.py` 的持久化验证语义：先从实际 JSON payload 得到 JSON-compatible expected value，再与落盘回读值比较。NaN/Inf 拒绝、已存在目标保护、原子替换和临时文件清理保持不变；profile schema、12 项矩阵、门限、求解器 runtime、生产默认值、PSSolver-Control 和 nematics3d 均不变。

本地门禁已经覆盖 tuple 到 JSON array 的回归测试、历史 helper/analyzer blob 身份、correct-layer 门禁以及完整 CPU suite。定向集合为 157 passed；完整 suite 为 3045 passed、7 个精确 CUDA-only deselect、8 subtests passed，零 failure、skip 和 xfail，用时 202.59 秒。当前只形成 qualification-only 本地提交；push 和新的 H100 recovery submission 都需要用户另行明确授权，且不得自动重试。
