# S-18 R4 等价性 helper 中立 profile provenance 更正

successor 99 以前向方式更正 R4 CPU 等价性 helper 的第二个证据设置问题，
不改写 successor 97 或 98，也不改运行时源码。

两个角色已使用相同工作目录后，profile 配置仍分别携带 `baseline` 与
`candidate` 角色及各自提交号。Plane 的 configuration identity 按合同包含这些
profile provenance 字段，所以身份仍然不同；状态张量和 refresh 进度没有出现
差异。最终 helper 用相同的中立 profile provenance 构造比较配置，同时在每个
child 结果中继续独立保存实际角色、提交、导入路径、环境和 Git 身份。

运行时实现仍是 `9cf54553b9f1bb2b94c2807c2e47adedeb415846`。本更正不改变
公开 API、数值算法、checkpoint schema、provider identity、门限、skip/xfail
或生产默认值，也未授权 push、发布、HPCC、Slurm、GPU 或 H100。
