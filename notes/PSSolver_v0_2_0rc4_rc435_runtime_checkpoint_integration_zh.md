# PSSolver RC4.3.5 runtime 与 checkpoint 集成

RC4.3.5 已完成本地实现与 CPU 验证，分类为 `PASS_RC4_3_5_LIFTING_RUNTIME_CHECKPOINT_INTEGRATION_CPU_CANDIDATE`。

本阶段把 RC4.3.4 的 fresh-only conditioning policy 正式分成两层：static lift、linear correction 和 homogeneous-remainder representation 继续属于 restart compatibility identity；conditioning policy、公式和“本次构造是否应用”进入 runtime metadata 与 checkpoint `run_provenance`，但不进入兼容 digest，也不改变 forward dynamics identity。

当前 writer 仍写 Plane checkpoint format v2，Plane forward identity 仍为 `plane.forward.v2`，static-lifting restart identity 仍为 schema 1。fresh-only policy 不需要版本提升；以后若真正改变 lifted forward representation，则仍必须按 RC4.3.2 合同提升 forward dynamics 与 lifting restart identity 版本，并重新做 CPU/H100 资格验证。

历史 format-v2 checkpoint 可以没有 fresh-construction provenance，也可以保留旧的 `fresh_initial_remainder_conditioning` 嵌套字段；只要 lift、linear correction、状态布局和其他科学 identity 完全相同，就允许精确恢复。未知 static-lifting schema 会在读取任何 tensor 文件之前拒绝；lift plan 或 correction 变化仍在 target mutation 前拒绝。format-v1 checkpoint 的既有显式升级规则没有改变。

真实文件级回归覆盖了 continuous 与 split/restart 逐字节一致、source/target fresh policy 不同仍精确恢复、旧 conditioning 字段兼容、缺失新 provenance 兼容、未知 lifting schema 预读取拒绝，以及不同 lift plan 预 mutation 拒绝。恢复路径构造时明确记录 `conditioning_applied=false`，恢复后的 checkpoint state 从未再次 conditioning。

本阶段没有开放 RC4.3.3 已关闭的 public lifted Plane capability，没有修改数值热路径、checkpoint tensor layout、PSSolver-Control、nematics3d 或生产默认值，也没有 push 或提交 H100。下一步是 RC4.3.6 installed-wheel CPU closure；localized policy 的 CUDA/H100 资格仍留给 RC4.3.7。
