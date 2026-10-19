# PSSolver RC6 pressure smoke 合同修正

analysis-only H100 Job `10901090` 已将旧聚合门禁唯一分解。float32 和 float64 的
tolerance 模式全部满足 termination、记录 residual、重算 residual 与 floor 合同；fixed
模式均精确执行 4 次且 pressure finite。唯一失败项是 qualification helper 期待不存在的
`fixed_iterations_completed`，而 installed-wheel、公共 pressure metadata、RC4/RC5 历史
合同和当前求解器均使用 `fixed_work_completed`。

本批次只修正 `benchmarks/run_rc6_contract_smokes.py` 中的 qualification-support 标签，并
将 tolerance/fixed-work 两个权威值集中为常量。新增 CPU float32/float64 真实 fixed-work
回归，要求 termination 为 `fixed_work_completed`、迭代数精确为 4、pressure finite。

旧 stage-producer 和 component-diagnostic 记录保持原样；本记录通过 forward-only successor
登记 helper 的前后哈希。旧 stage-producer 审计测试仅增加对这份精确 successor 登记的解析，
不会接受未登记的源文件变化。没有修改 runtime、solver、数值实现、pressure policy、阈值、
公共 API、PSSolver-Control、nematics3d 或生产默认值，也没有增加 skip/xfail。

本修正没有 H100 提交权限。完成本地 CPU 门禁后，只能准备新的 hash-frozen support 和
stage plan；正式 RC6 H100 qualification rerun 仍需单独明确授权。
