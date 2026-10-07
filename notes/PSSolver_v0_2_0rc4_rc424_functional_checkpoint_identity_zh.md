# PSSolver RC4.2.4 functional checkpoint derivative identity 与显式状态迁移

## 结论

RC4.2.4 为 Periodic 和 Channel functional checkpoint 引入了 v3 分层兼容身份。exact restart 现在同时要求 `forward_dynamics`、`derivative_dynamics`、`state_layout` 和 `backend_restart` 一致，而 requested device、snapshot、output policy 与运行 provenance 不参与兼容 digest。

本阶段关闭了冻结矩阵中的 M13、M14，并保持 M15/M16 的 fail-closed 边界。provisional 或 derivative-incompatible 的旧 functional checkpoint 不再直接进入当前 reader；调用者必须显式选择 source release generation，并把迁移结果写入新目录。已资格化的 Periodic post-repair v2 reader 继续保留。迁移记录在读取 tensor 前完成 schema 验证，source 目录不被改写，也不会把 state-only migration 误报为旧轨迹的 exact continuation。

## Periodic functional

- 当前 bridge format 为 v3；
- rc1 `0.1-provisional` 位于 Hermitian state repair 之前，只允许用当前 projector 从 physical Q 重建 spectral Q；
- 该路径明确标记为 `state_only_migration`、`trajectory_equivalent=false`；
- rc2/rc3 post-repair v2 checkpoint 保留已资格化的直接 reader；在完整旧身份匹配时也可显式升级为 v3，并标记为 `exact_legacy_upgrade`；
- 当前 functional v3 虽继续复用已密封的 production-v2 tensor carrier，但 production exact restore 不接受 v3 functional bridge，避免跨 family 静默恢复。

## Channel functional

- 当前 bridge format 为 v3；
- derivative identity 冻结 custom implicit pressure adjoint、transpose action、even-periodic Nyquist projection 和 zero-start transpose guess；
- rc1/rc2 的偶数 periodic axis 位于 rc3 pressure-transpose 修复之前，只允许 state-only migration；
- rc1/rc2 的奇数 periodic axis 已证明该修复不适用，可显式 exact upgrade；
- rc3 已含修复，在 forward identity 匹配时可显式 exact upgrade；
- migration provenance 还冻结 periodic axis size，因此不能通过伪造一个 `migration_kind` 字段改变 exact/non-exact 判定。

## API 与范围

stable functional API 版本仍为 `1.0`；本阶段只为 checkpoint bridge 增加了显式 `migrate_legacy_checkpoint(...)` 能力，并提供 Periodic/Channel 的非原地便利入口。模型方程、timestep 热路径、生产默认 runtime、PSSolver-Control 和 `nematics3d` 均未修改，也没有运行 H100。

由于 Provider 的 functional checkpoint identity 和公开 bridge 能力发生了变化，PSSolver-Control 的 installed-wheel、checkpoint 和 gradient 累计资格必须在后续独立阶段重新执行；RC4.2.4 本身不把此前 Control 资格自动延伸到新格式。

## 下一步边界

RC4.2.4 完成后可以规划 RC4.2.5：用 installed wheels 执行跨版本 CPU 矩阵、continuous/restart 等价、pre-mutation negative gates 和完整 CPU suite。当前没有授权实施 RC4.2.5、修改 Control、提交 H100、自动 merge、改变默认值或发布。
