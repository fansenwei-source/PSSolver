# PSSolver v0.2.0rc6 源码冻结准备

RC6-1 至 RC6-7 已按既定顺序完成源码实现和相应 CPU 证据。该步骤把
Provider 包版本从 `0.2.0rc5` 切换为 `0.2.0rc6`，冻结公开发布说明，并登记
独立 PSSolver-Control guard-order 回归证据的精确源码与记录提交。

冻结源码完整 CPU suite 已通过：`3936 passed, 7 deselected, 8 subtests
passed`，没有 failure、skip 或 xfail。七个 deselect 精确对应既有 CUDA-only
node ID；本轮没有把 CUDA 结果当作 CPU 证据。

这不是累计资格 PASS。后续门禁顺序为：fresh installed-wheel CPU、真实旧版
checkpoint compatibility/tamper、Provider
identity 影响审计、必要时独立 Consumer 迁移与 installed-wheel 资格，以及一次
另行明确授权的单卡 H100 累计验证。

本步骤不 push，不创建或移动 tag，不发布 GitHub Release 或 PyPI，不提交
H100，不修改 PSSolver-Control 或 `nematics3d`，不开放 lifting，不改变生产
默认值，也不授权自动重试。
