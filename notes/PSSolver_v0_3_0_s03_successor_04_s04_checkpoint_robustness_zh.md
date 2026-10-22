# PSSolver v0.3.0 S-04 checkpoint 健壮性候选

本地分支 `fix/v0.3-s04-checkpoint-robustness` 已在实现提交
`468e06710e1b4881a557f0228dcb16be38791094` 中处理 V-22、V-16、V-13、
V-14 和 C2-17。

核心约束是：固定事务开始时的真实父目录；NFS 回退以打开的目录描述符和
一次性所有者标记持有占位；发布后的父目录同步失败使用独立 durability
异常；只读 Channel 旧 checkpoint 的显式升级只修改私有 staging 副本。

checkpoint schema、checkpoint payload、provider identity、求解器方程、数值算法
和生产默认值均未改变。

本地门禁已经通过：定向测试 `41 passed`；完整 CPU suite 为
`4340 passed, 1 skipped, 7 deselected, 8 subtests passed`，唯一 skip 是缺少
可选 `nematics3d`；N-1 模拟矩阵的 528 个 errno/statfs/backend 选择组合为
`0 mismatch`，五组发布语义以及三个 umask 下的 fast/fallback 字节一致性均通过。
资格化副本只为适配 S-04 已删除的私有 helper 和新的 0700 owner-marker claim，
原始脚本哈希被保留；它不是产品源码。

由于本批修改了 NFS fallback，仍须由用户单独授权一次 HPCC 真实 NFS 复测，
之后才能提交 R-0010 给 Claude 验收。当前分类为
`PASS_LOCAL_V0_3_S04_CHECKPOINT_ROBUSTNESS_PENDING_REAL_NFS`。
