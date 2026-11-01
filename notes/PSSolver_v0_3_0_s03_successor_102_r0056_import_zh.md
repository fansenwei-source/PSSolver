# successor 102：导入 R-0056

`R-0056` 在提交 `990349056290936e497c449a6abf2592c47cf448` 上独立复核
通过，S-18 的第二批 R4 已完成。运行时实现提交仍为
`9cf54553b9f1bb2b94c2807c2e47adedeb415846`；R-0056 没有登记表更新或
新发现，聚合项 R1-R9 继续保持 `deferred`，等待 R5 和 R1 各自完成。

R-0054 的 R4 清单中有五个标为“本批修改”的文件最终有意保持不变：

- `pssolver/integrator.py` 保留旧积分器独立的 `step_count`、
  `refresh_count` 和刷新推进逻辑。它是兼容性与对照边界；将其接入新的
  state-backed 私有原语会扩大架构依赖，却不会消除已资格化 runtime 的重复。
- `pssolver/integrators/scalar_reference.py` 保留独立 reference/oracle 实现，
  使其继续作为共享 stepping 实现之外的对照，不把 oracle 与被测实现合并。
- `pssolver/workflows/channel_beris_edwards.py` 的 progress/restart 集成已经通过
  本批修改的 runtime adapter 委托；wrapper 本身无需改动。
- `pssolver/workflows/periodic_checkpoint.py` 的发布与读取字节合同已由 R2 冻结，
  R4 没有改变其 progress 表示或调用图，因此不做无收益改写。
- `pssolver/workflows/plane_checkpoint.py` 同样保持 R2 后的格式、读取和发布边界；
  R4 的 progress 恢复由 runtime/state owner 完成，不需要修改该文件。

这五项不是留给 R5 或 R1 的隐含工作；它们是 R4 完成后的明确冻结边界。
另外，`step_program` 现在会在 `begin_generation()` 前读取只读的
`state.progress.dt`。R-0056 已核对该时序变化不影响状态、报错、计数或身份，
因此只作信息记录。

复核确认私有原语、MRO、操作顺序、异常文本、8 family × 5 个刷新设置 ×
20 步逐位等价、2463 个稳定 writer 文件、checkpoint/restart、两套全量 CPU、
外部路径屏蔽、干净 wheel 和 PSSolver-Control main 门禁全部通过；没有触发
Control 迁移、真实 NFS 或 GPU 门禁。

本提交只导入复核结果，不修改运行时、打包、公开接口、数值实现、checkpoint
schema、provider identity、门限、skip/xfail 或生产默认值，也不授权 push、
tag、Release、PyPI、HPCC、Slurm、GPU 或 H100。
