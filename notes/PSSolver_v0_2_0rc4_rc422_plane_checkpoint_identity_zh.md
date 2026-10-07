# PSSolver RC4.2.2 Plane checkpoint 分层身份与旧版升级

## 结论

RC4.2.2 已把 Plane workflow checkpoint 从只保存 opaque runtime SHA 的 format v1 迁移到 format v2。新格式显式保存 `forward_dynamics`、`state_layout` 和 `backend_restart` 三层兼容身份；requested device、初始化来源及输出策略仍可作为 provenance 记录，但不再错误决定 exact restart 兼容性。

本阶段只改变 Plane checkpoint reader、writer 和 restore gate。Periodic、Channel、functional checkpoint、PSSolver-Control、`nematics3d`、生产默认 runtime 和 timestep 热路径均未改变，也没有运行 H100。

## 当前 v2 合同

- 新 writer 只写 format v2；
- v2 metadata 同时保存完整分层 identity 和 canonical SHA-256；
- header reader 在读取 tensor 前验证版本、runtime path、identity schema 和 digest；
- restore 在首次 target tensor 写入前验证 forward dynamics、state layout、backend restart 和 lifting；
- runtime path 仍必须相同，没有放宽 cross-runtime restart；
- `device` 与 `fresh_initial_remainder_conditioning` 不属于兼容 digest；
- `tf32`、Q-gradient reuse、模型、几何、边界、数值离散、timestep、spectral refresh、prescribed lift 与 linear correction 仍属于动力学身份。

## v1 升级合同

format v1 保持可读，但不能直接 restore。调用者必须提供明确的 source `PlaneBerisEdwardsRunSpec` 与 rc1/rc2/rc3 generation；升级器先通过冻结 registry 和旧 opaque digest 验证来源，再决定是否允许写一个新的 v2 目录。原 v1 目录不会被原地修改。

RC4.1 改变了偶数 periodic grid 的 Nyquist 语义，因此任一 periodic axis 为偶数时，rc1–rc3 v1 checkpoint 均不能被宣称为 exact restart。只有两个 periodic axis 都为奇数、可由 source run spec 直接证明该变化不适用时，才允许升级。rc1 lifted checkpoint 也因旧 lifting dynamics 不兼容而 fail closed。

## M01–M06 结果

- M01：只改变 device token，接受 exact restart；
- M02：只改变 fresh initializer 或 fresh remainder conditioning，接受；
- M03：prescribed lift 或 linear correction 不同，在目标状态写入前拒绝；
- M04：rc1 lifted checkpoint 拒绝 exact upgrade；
- M05：rc1/rc2/rc3 偶数 periodic grid 在 tensor load 前拒绝；
- M06：rc1/rc2/rc3 双奇数 periodic grid 经显式升级后 exact restart 通过，且 source 目录逐文件 SHA-256 不变。

## 下一步边界

RC4.2.2 完成后可以规划 RC4.2.3：为 Periodic 与 Channel production checkpoint 引入各自的 forward-dynamics version 和 legacy reader。当前没有授权直接实施 RC4.2.3、修改 functional derivative identity、修改 Control、提交 H100、自动 merge、改变默认值或发布。
