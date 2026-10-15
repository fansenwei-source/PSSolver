# PSSolver v0.2.0rc5 G-1.0 源码冻结准备

当前工作队列 W0、W1、R5-1 至 R5-13 已完成本地实现和既定 CPU
证据。G-1.0 将包版本冻结为 `0.2.0rc5`，新增当前 release record，且把
`v0.2.0rc4` 的测试改为明确的历史合同；它不改写 rc4 的 tag、Release、
历史记录或发布产物。

这一步只表示“可以开始对冻结源码做资格验证”，不表示 G-1 已通过。
接下来必须依次完成：冻结源码全量 CPU、冻结 installed-wheel CPU、真实
rc1–rc4 checkpoint 矩阵、Provider identity 影响审计、必要时独立
PSSolver-Control installed-wheel 资格，以及一次另行明确授权的单卡 H100
累计资格。H100 必须覆盖 CPU↔CUDA checkpoint、包含 `"medium"` 的 TF32
调用方状态、标准数值非回退、内存与性能门禁。

本批不 push，不创建 tag，不发布 GitHub Release 或 PyPI，不修改
PSSolver-Control 或 `nematics3d`，不开放 lifting，不改变生产默认值，也不
授权自动重试。
