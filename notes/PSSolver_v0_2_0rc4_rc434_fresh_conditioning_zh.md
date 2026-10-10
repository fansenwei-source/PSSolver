# PSSolver RC4.3.4：显式 fresh-state conditioning 策略

RC4.3.4 已把原先隐藏在 lifted Plane 初态构造中的 whole-slab sine taper 拆成显式、不可变、可记录的 fresh-only 策略。默认策略现在是 `strict_no_automatic_conditioning_v1`：只计算 `physical - lift`，不再悄悄改写用户请求的初态。

显式 opt-in 候选 `localized_quintic_wall_transition_v1` 只在距最近墙面小于 `transition_width` 的区域使用五次 smoothstep；距离大于等于 width 的内部网格值通过 `torch.where` 直接选择原 remainder，因此逐字节保持不变。width 使用几何物理长度单位，且必须满足 `0 < width <= L_wall/2`。旧行为被保留为 `legacy_whole_slab_sine_v1`，但只能显式请求，不再是任何默认路径。

conditioning metadata 包含 schema、requested/effective policy、公式、width、单位、字段、分量及 `fresh_only`。它属于 fresh construction 的复现 provenance，不属于 forward dynamics 或 restart compatibility digest。应用入口在 restart 构造时显式设置 `apply_fresh_initial_conditioning = false`；恢复的 homogeneous remainder 不会被再次调理。

本阶段没有改变 RC4.3.3 的 public fail-closed 限制，也没有开放非零 prescribed-Q Plane 执行；没有修改 checkpoint 格式、求解器热路径、PSSolver-Control、nematics3d 或生产默认值。localized policy 当前只是 CPU candidate，CUDA 确定性和 H100 资格仍留给 RC4.3.7。
