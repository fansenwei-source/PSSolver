# PSSolver v0.2.0rc4 RC4.5.5 public validation、capability 与 diagnostics 实现记录

RC4.5.5 按冻结合同关闭 B13、B15 和 B17。四个 public application compiler 现在共享严格 device token 语法；两个声明 iterative pressure solver 的 Channel family 共享 tolerance、iteration、fixed/max、warm-start 类型与精确 key 校验。所有校验均发生在 output、runtime 或 CUDA allocation 之前，compiler normalization 分别记录 requested 与 resolved device，allocated identity 仍由 runtime metadata 记录。

Capability catalog 继续由 live compiler registry 生成组合真值，并显式登记 Periodic `no_slip_velocity` 因无物理 boundary face 而等价于 effective `periodic_continuation`。Periodic compiler 同时记录实际 requested policy、effective policy 与 equivalence reason。已知 binding、lowering 和 option-validation 边界的内部异常统一翻译为带 machine code 的 `PublicSimulationCompilationError`，原异常保留为 cause。

Plane complete-stress、legacy Channel、Periodic complete-stress 和 Channel complete-stress 四条 workflow 均在 `metadata.json::diagnostics` 写 schema version 1 envelope。它包含 application/runtime 身份、enabled/count/steps、artifact paths、按 structured dtype 顺序排列的 payload schema，以及覆盖全部字段的 units。已有 `diagnostics.npy`、`diagnostics.csv`、字段名、dtype 和 payload serialization 未改变；0.2 系列的顶层 `npy`/`csv` alias 继续保留。disabled diagnostics 仍声明 schema 与 units，但 count/steps 为空且 artifact path 为 null。

本阶段不修改 B11 scoped TF32，不修改模型方程、时间推进、谱变换、pressure algorithm、PSSolver-Control、`nematics3d` 或生产默认值；没有运行 H100、push、merge 或发布。CPU 验证完成后才进入 RC4.5.6。
