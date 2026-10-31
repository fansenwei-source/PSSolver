# R-0049 S-17 返工复核导入

`R-0049` 独立验证了 S-17 的三项返工：Control 的可达历史核对、Control main
迁移披露，以及 PSSolver 的 R3/R6 Consumer 边界设计。`V-56`、`V-57`、
`V-58` 均推进为 `verified`；修复身份分别保持
`7e719bbbcd961f54532756ca8929fac3d441b995`、
`7e719bbbcd961f54532756ca8929fac3d441b995` 和
`d98472aa975776fcd78f95b133d2abb747bce83b`，`verified_by=R-0049`。

`D-0017` 已确认保全方式 A：后续 Control 推送草稿须同时列出 main 的非强制
快进，以及用于保全 `48dd4d3…` / `97d3c06…` 的
`migration/pssolver-v0.3-s09-execution-scope` 新分支；
`migration/pssolver-v0.3.0-main-sync` 本身不推送。研究分支不在迁移范围内，
门禁允许它被独立研究会话推进，但其他远端引用必须保持快照一致。

本 successor 只导入复核结果，不修改运行时、打包、公开接口、数值实现、
checkpoint schema、provider identity、门限、skip/xfail 或生产默认值，也不授权
任何 push、tag、Release、PyPI、HPCC、Slurm、GPU 或 H100 操作。R3/R6 实现只在
后续推送另行授权并完成后开始。
