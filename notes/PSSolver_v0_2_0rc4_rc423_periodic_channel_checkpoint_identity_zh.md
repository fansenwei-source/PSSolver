# PSSolver RC4.2.3 Periodic / Channel production checkpoint 分层身份

## 结论

RC4.2.3 已为 Periodic production 与 complete-stress Channel production checkpoint 引入显式的 `forward_dynamics`、`state_layout` 和 `backend_restart` 三层兼容身份，并为 rc1–rc3 的旧格式提供 fail-closed、非原地的显式升级路径。冻结计划中的 M07–M12 已全部关闭。

本阶段没有改变 timestep 热路径、模型方程、数值离散、生产默认 runtime、PSSolver-Control 或 `nematics3d`，也没有运行 H100。requested device 和 output policy 仍可作为 provenance 保存，但不参与 exact restart 兼容 digest。

## Periodic production

- 当前 writer 只写 format v3；旧 format v1/v2 不允许直接从 production workflow restore；
- rc1 `full_complex` 位于 Hermitian repair 之前，必须在 tensor load 前拒绝 exact restart；
- rc1 `hermitian_half` 已证明 repair 不适用，可在验证旧 opaque identity 后显式升级到 v3；
- rc2/rc3 post-repair checkpoint 可显式升级到 v3；
- 升级总是写入新目录，source checkpoint 的逐文件内容保持不变；
- 已有 qualified functional bridge 的 v2 carrier 与直接恢复能力暂时保留，functional derivative identity 和旧 functional state migration 归 RC4.2.4。

## complete-stress Channel production

这里的 Channel 指 `channel_complete_stress` workflow，而不是更早的 P7 `channel_checkpoint.py` runtime。

- 当前 writer 只写 format v3；
- rc1 transport 为 v1，rc2/rc3 transport 为 v2；
- rc1 偶数 periodic axis 使用 RC4.1 之前的 Nyquist forward dynamics，升级器必须在 tensor load 前拒绝；
- rc1 奇数 periodic axis 可直接证明 Nyquist 变化不适用，因此允许显式升级并 exact restart；
- rc2/rc3 只存在 identity schema 差异，允许升级并 exact restart；
- pressure guess、Q、velocity 与 pressure 的 state-layout / backend restart 身份均在目标写入前验证。

## M07–M12

- M07：Periodic rc1 pre-repair `full_complex`，拒绝 exact restart；
- M08：Periodic rc1 `hermitian_half`，显式升级后 exact restart 通过；
- M09：Periodic rc2/rc3 post-repair，显式升级后通过；
- M10：Channel rc1 偶数 periodic grid，在 tensor load 前拒绝；
- M11：Channel rc1 奇数 periodic grid，显式升级后通过；
- M12：Channel rc2/rc3，显式升级后通过。

所有接受路径都验证了 source 目录不被原地修改；所有拒绝路径都验证了不会读取 tensor payload、不会部分构造升级目录，也不会修改目标 runtime。

## 下一步边界

RC4.2.3 完成后可以规划 RC4.2.4：专门处理 Periodic / Channel functional checkpoint 的 derivative-dynamics identity、M13/M14 和显式 state-only migration。当前没有授权修改 Control、提交 H100、自动 merge、改变默认值或发布。
