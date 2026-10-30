# S-16 R-0044 返工前向登记

R-0044 独立确认 V-50、V-51、V-52 的修复有效，三项保持既有
`fixed_in` 并更新为 `verified_by=R-0044`。本记录不改写 successor 72、73
或其他历史记录。

R-0044 新发现 V-53：`tests/test_v0_4_s16_r0043_rework.py` 曾在
PSSolver_sync 的 R-0043 结果文件存在时才执行哈希断言，在新机器上会静默跳过。
`05513a4132b1eb38e9bdff825f98a31b276c2932` 删除了该外部路径常量和条件分支；
successor 72 内冻结的结果哈希继续由仓库测试核对，完整的本机外部证据仍由
D-0015 只读核对工具的 C023 检查。

对 `tests/**/*.py` 的复查没有发现其他按 `/home/fansenwei` 外部证据路径是否
存在而改变 pytest 结果的分支。仓库 fixture、临时目录、测试生成物和可执行文件的
正常存在性检查不属于该问题。

本次只改记录和测试，不修改运行时、公开接口、数值算法、checkpoint schema、门限、
skip/xfail 或生产默认值；未获授权执行 push、tag、Release、PyPI、HPCC、Slurm、GPU
或 H100 操作。V-53 先登记为 `claimed_fixed`，等待下一轮独立复核。
