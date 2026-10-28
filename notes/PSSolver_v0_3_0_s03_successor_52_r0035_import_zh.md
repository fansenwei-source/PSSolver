# R-0035 S-09 结案复核导入

日期：2026-10-27
路线图步骤：S-09
分类：`PASS_R0035_S09_CLOSURE_REVIEW_IMPORTED`

本提交只导入 `R-0035` 的独立复核结论，不修改运行时、打包、资格支持代码、冻结计划、
作业、分析器、阈值或生产默认值，也不授权 push、HPCC、Slurm、H100、tag 或 Release。

`R-0035` 独立确认：S-09 v2 重算使用作业 `10911538` 的真实冻结 support plan；重算产物
SHA-256 为 `1757374241e033c77a5fbe0b31df29e6f33a0dcae3e777c4ae45754d5927df01`，
逐字节复现；记录的失败严格只有 `D-0013` 覆盖的两个 R512 float32 Channel
`peak_reserved` 分组；五个负向控制均失败关闭；fail-fast 分析器只停在第一个
`channel_legacy` 失败，第二个 `channel_compiled_v2` 失败由不中断重算发现。

因此 `V-45` 从 `claimed_fixed` 更新为 `verified`，`verified_by=R-0035`。实际包含修复的
提交是 `e052f5b87e1424fbd8d9bd097a893469f9c72658`；祖先 `719ae760…` 不包含该修复，
不再作为 `fixed_in`。

原 S-09 结案记录仍保存当时 `V-45=claimed_fixed` 的历史快照；本 successor 只向前登记
状态变化，不改写原记录。
