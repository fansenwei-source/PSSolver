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

本地资格已通过：重点集合 `201 passed`；完整 CPU suite 为 `4518 passed`、允许的
optional nematics3d `1 skipped`、精确 CUDA-only `7 deselected`；installed-wheel
公共合同为 `5 passed`，`pip check` 通过且没有源码遮蔽。wheel 不含旧 control 或仓库
qualification helper，并保留所需 experimental runtime。

第一次完整 suite 的三个失败只来自历史记录测试仍假定当前没有待验收条目；它们已改为
精确识别本批六个 `claimed_fixed`，首次日志保留在证据归档中。运行时资格源仍是
`b7cf4ca02095e8b41ed720809880628d2375cfbb`。下一步封存本地证据并创建 `R-0017`
交 Claude 独立复核。未执行 push、HPCC、Slurm、GPU 或 H100。

证据归档为
`/home/fansenwei/pssolver_v0_3_s07_local_qualification_20261022_v1`，清单共
`2225` 项，`checksums.sha256` 自身 SHA-256 为
`89e6b4d69964ea75e09c3f38223d50ff2839c24cc4d2d1ea4c4a9ed8f6a060ac`。
两个 D-0008 硬链接环境克隆已删除，来源、导入路径与 `sys.path` 保存在归档中。
