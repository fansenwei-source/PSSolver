# S-04 V-25 返工候选

提交 `44a6fc3d88d94ccf939e78e23be3ffbac542def5` 修复 R-0010 新发现的 V-25。
NFS 兼容发布路径在占位目录认领失败后先检查最终目标：目标不存在时，说明 `mkdir`
明确失败且未开始 rename，代码清理完整 staging 并抛公共 `CheckpointIOError`；只有目标
存在但无法认领时才保留 ambiguous 状态和诊断用 staging，并在异常 details 中写出路径。

ENOSPC、EDQUOT、EROFS、EACCES 与 EIO 五种明确失败均有无残留回归测试。原 V-13
lost-reply 恢复、V-14 外来替换保护、V-16 durability 分类、V-22 稳定父目录和 C2-17
只读升级语义不变。checkpoint schema、checkpoint bytes、provider identity、求解方程、
数值算法、阈值和生产默认值均未改变。

V-25 目前为 `claimed_fixed`，此前五项也继续保持 `claimed_fixed`。必须先由 R-0011 做
短复核；复核 PASS 后才能按 D-0009 起草真实 HPCC NFS 授权。本提交不授权 push、HPCC、
Slurm、GPU 或 H100。
