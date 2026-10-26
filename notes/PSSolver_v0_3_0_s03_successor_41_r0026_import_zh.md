# R-0026 导入记录

R-0026 独立复核通过 S-09 recovery 的 qualification-support 改动。`V-39` 与 `V-40`
均绑定修复提交 `72286694c43ee15e5859c0658c989431404fe505`，推进为 `verified`，
`verified_by=R-0026`。本次导入不修改运行时、wheel、冻结的 plan/job、阈值或案例；
`NEW-14` 继续保持 `open`，S-09 仍须等待一次单独授权的恢复 H100 作业及其结果验收。

R-0026 明确允许依据 D-0009 引用已冻结的恢复作业输入身份起草授权文字，但它本身不授权
HPCC 写入、Slurm/H100、push、tag 或 Release。
