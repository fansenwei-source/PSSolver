# S-18 R-0052 范围返工

首版 S-18 设计把四批的高层目标和主要文件写清，但没有把静态扫描发现的所有
checkpoint、完整性 helper、谱刷新逻辑和组合专用模块逐项处置。实现提交
`f5dbdbef43d1aa16e9c7638093df5c5c5ae58170` 增加权威的逐文件分类表：每个文件恰好
归入“本批修改”“只读冻结”或“范围外”，并写明理由。

返工同时明确：每批都跑完整 PSSolver-Control main CPU suite；R4 的两条 Control
Functional runtime 路径只读冻结；R2 只有改变 staging、fsync、rename/renameat2、
placeholder、清理、symlink/device 判定或其调用图时才必须另行授权真实 NFS 复测。
代码扫描棘轮会拒绝未分类的 checkpoint/integrity/spectral-refresh 模块。

`V-60` 现为 `claimed_fixed`，修复身份为上述提交，等待 `R-0053` 独立复核。
`D-0018` 仍为 `proposed`，所以没有开始任何 S-18 运行时实现。本 successor 不授权
push、tag、Release、PyPI、HPCC、Slurm、GPU 或 H100。
