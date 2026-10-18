# PSSolver v0.2.0rc6 H100 stage producers

本地 qualification-only 实现已经补齐 RC6 单次 H100 累计资格的 11 个有序阶段：环境、CUDA-only、RC6 公共合同 smoke、TF32、当前 checkpoint、历史 checkpoint、八族数值比较、192 项 profile、Consumer、聚合和最终 analyzer。

每个阶段使用 shell-free argv，stage-plan 冻结时绑定 Python 可执行文件和 helper 的绝对路径及 SHA-256；每个正式 JSON 均拒绝覆盖、禁止 NaN/Inf，并执行原子写入和回读。失败会阻止后续阶段。

本批次没有修改求解器 runtime、公共 API、checkpoint schema、Functional API 或生产默认值，也没有修改 PSSolver-Control 和 nematics3d。

当前仅完成本地 producer 与 synthetic 门禁。实际 archive、wheel、三个 installed-wheel venv、非 shadowing support 树和具体 stage-plan 必须在未来获得 push 与单次 H100 明确授权后另行构建和冻结。本记录不构成 H100 PASS，也不授权 push、提交、tag 或发布。
