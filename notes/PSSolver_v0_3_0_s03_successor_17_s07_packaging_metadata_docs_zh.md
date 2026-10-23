# PSSolver v0.3 S-07 打包、元数据与文档候选

S-07 的本地实现源为 `b7cf4ca02095e8b41ed720809880628d2375cfbb`，基于
`dev/0.3@98742b6517ab43f519ff45663e6524bc96a41cd2`。本批处理
`A1-07`、`S1-15/NEW-13`、`S1-04`、`OBS-4`、`B30` 与 `V-26`；六项在
独立复核 `R-0017` 前保持 `claimed_fixed`。

候选 wheel 不再包含旧的 `pssolver.control`，也不包含仓库内部的
`pssolver.experimental.stage_*` 与 H100 qualification helper，但保留 PSSolver 本身
所需的 experimental runtime。公共 `Simulation` 运行元数据现在持久化 compilation
identity 和完整 normalization。新 Plane 结果使用 schema 2 的稳定 application 身份；
旧 schema 1 脚本标签仍可读取和用于兼容续算。

checkpoint 发布成功且 durability 确认后，会通过结构化日志诊断报告使用了
`rename_noreplace` 或 NFS owned-placeholder 后端；该信息不进入 checkpoint/provider
identity。Functional checkpoint import 结果显式携带 `sealed_sha256` 或
`unsealed_legacy`。需要封印保证的调用者必须检查前者；SHA-256 不是 MAC 或数字签名。

Robin 上下壁残差是独立的 wall-law boundary gate，但不是完整算子的独立 correctness
gate；内部 Helmholtz、evolution 和 restart 仍有各自门禁，Robin 公共执行没有重开。

本记录尚待最终本地 full CPU 与 installed-wheel 资格收口，随后创建 `R-0017` 交由
Claude 独立复核。未执行 push、HPCC、Slurm、GPU 或 H100。
