# S-18 R4 等价性 helper 工作路径更正

successor 98 以前向方式更正 R4 的 CPU 等价性 helper，不改写 successor
97，也不改任何运行时源码。

第一次父提交对候选的资格尝试分别使用了带 `baseline` 和 `candidate`
角色名的工作目录。运行状态和 refresh 进度没有出现差异，但工作目录进入了
configuration identity，因此该尝试不能作为逐位身份等价证据。helper 现在让
两个角色依次使用并清理同一个确定性工作目录。失败尝试原样保留，最终证据从
修正后的 helper 重新生成。

运行时实现仍是 `9cf54553b9f1bb2b94c2807c2e47adedeb415846`；本更正不改变
公开 API、数值算法、checkpoint schema、provider identity、门限、skip/xfail
或生产默认值，也未授权 push、发布、HPCC、Slurm、GPU 或 H100。
