# R-0010 S-04 预审结果导入

R-0010 的独立复核分类为 `REWORK_REQUIRED`，没有发布阻断项。V-22、V-16、
V-13、V-14 与 C2-17 五项继续保持 `claimed_fixed`，没有提前改为 verified。

新发现 V-25 已登记为 `registered`、目标 `0.3`：NFS 兼容发布路径上的占位目录
`mkdir` 明确失败时，候选代码仍尝试认领不存在的目标，继而把结果误分为 ambiguous，
并保留完整 staging。返工要求是在认领失败后先 `lstat` 目标；目标不存在时清理 staging
并抛明确的 `CheckpointIOError`，只有目标确实存在但无法认领时才保留模糊状态。

本次提交只是 evidence-only 导入，不修改运行时、checkpoint schema、provider identity、
阈值、skip/xfail 或生产默认值，也不授权 push、HPCC、GPU 或 H100。
