# successor 105：R5 后的冻结 S-09 proxy 测试兼容

R5 有意重构了生产 `_solve_pressure` 的 Python 源码，因此冻结的 S-09
撤回 proxy 测试不能再把当前方法误当成它绑定的历史候选方法。本 successor
只修改测试：从仍可达的历史提交读取并核对原候选方法，随后执行原样未改的
冻结 proxy，并恢复当前 R5 方法。

这不是“对象缺失就跳过”的分支，不新增 skip/xfail，也不修改任何冻结
benchmark、运行时、打包、数值、门限或生产默认值。本记录不授权远端、
HPCC、Slurm、GPU 或 H100 操作。
