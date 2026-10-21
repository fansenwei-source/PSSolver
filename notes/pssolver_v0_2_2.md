# PSSolver v0.2.2 本地发布源准备

日期：2026-10-20

分类：`PASS_V0_2_2_LOCAL_RELEASE_SOURCE_PREPARATION`

本地分支 `release/pssolver-v0.2.2` 以已经推送并通过 R-0002 独立复核的
`701991a0af6cab1cf0952f76e16c430f2c2613ac` 为直接父提交。v0.2.2 的运行时
改动范围严格限于 V-17：checkpoint 的直接父目录为符号链接时，NFS 兼容发布
路径使用解析后目录的设备身份。除此之外只提升公开版本号并补充发布文档。

R-0002 已独立验证源码和本地 installed wheel 的 Periodic、Channel checkpoint
导出、逐位 round trip、强制 fallback、真实 ext4→tmpfs 父目录符号链接以及真实
跨文件系统拒绝。V-17 已在统一问题登记表中标记为 `verified`。

V-22 与 V-23 是从 v0.2.0 起即存在的低严重度边界问题，已经登记给后续
v0.2.x；本发布不扩大范围去修复它们。

当前提交只准备发布源，不代表正式发布。以下事项仍待后续阶段：

- 从干净源码归档构建并验证 wheel/sdist；
- installed-wheel 发布门禁；
- 经单独授权，在 HPCC 登录节点对真实 NFS 父目录符号链接进行小型复测；
- push、annotated tag 与 GitHub Release。

本路径解析修复不需要重新运行 H100。未授权 PyPI 发布。

本地发布源完整 CPU 门禁已经通过：4179 passed、7 个精确 CUDA-only
deselect、0 failed、0 skipped、8 subtests passed。
