# PSSolver RC6 pressure component diagnostic

RC6 recovery Job `10900775` 已通过 environment、7 个 CUDA-only tests 和 contract
pytest 的 36 项测试，但聚合的
`pressure_floor_fixed_and_tolerance_modes_on_cuda` 返回 `false`。旧 helper 没有在
抛出异常前保存 dtype、mode、termination reason、记录 residual、重算 residual 或
finite 子谓词，因此不能唯一归因。

本记录新增 analysis-only H100 diagnostic。它逐项复用失败 smoke 的 shape、lengths、
seed、friction、viscosity、float32/float64 floor，以及 tolerance/fixed iteration 参数。
每个 dtype/mode 都写入独立的原子 JSON；真实 gate FAIL 仍作为完整诊断正常结束，只有
环境、构造或观察异常才返回非零。这样可以区分合同判定失败与诊断程序自身失败。

版本化 Slurm 入口只运行这项 analysis-only diagnostic。它在诊断前强制完成 node-local
`/tmp` 的 `RENAME_NOREPLACE` 检查和 CUDA 12.1 工具链检查，并把 `TMPDIR` 固定到独立的
node-local 目录；它不调用 RC6 stage plan、正式 analyzer，也不写 `COMPLETE`。

本批次没有修改求解器、pressure policy、阈值、公共 API、PSSolver-Control、nematics3d
或生产默认值，也没有增加 skip/xfail。它没有资格 PASS 含义。push 与 analysis-only
H100 Job 都需要新的明确授权，且不能把未来诊断 Job 当作 qualification rerun。
