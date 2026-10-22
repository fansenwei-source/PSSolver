# S-06：导入 R-0015 独立复核

R-0015 的分类为 `REWORK_REQUIRED`。它确认 `V-28` 已阻止发散状态被标为 acceptable，
并再次通过 `C1-05`、`C2-18` 与 `V-27`；但最后一个下限内递推状态并不总是最佳回滚点，
且严格检查从未发生回滚的正常 fixed-work 完成引入了 `V-29`。

本次登记 `V-29`（low，S-06），五项既有问题仍保留 `claimed_fixed`，不提前标为 verified。
返工合同要求保存下限内递推残差最小的状态；只有实际发生回滚时才严格重算并检查 represented
residual。正常完成维持 S-06 之前的接受语义。完整 2160 次扫描和 360 次 floor-crossing 扫描都必须
报告 fail-closed 次数且为零。

本提交只导入证据、更新登记表和绑定测试，不修改运行时，不授权 push、HPCC、GPU 或 H100。
