# PSSolver v0.3.0 S-10 资格预检历史测试修正

首次干净完整 CPU 门禁得到 `4772 passed / 1 skipped / 7 deselected / 8 subtests passed / 11 failed`。十一项失败全部来自历史证据测试把当前登记表状态或当前文件哈希固定为旧值；运行时、数值、科学、阈值和公开 API 测试均未失败。

本次修正只调整十个历史绑定测试：历史记录自身继续按原值校验，当前状态改为通过已有 successor 链解析；V-33 的 `claimed_fixed` 状态被显式识别。S-08 的 checkpoint 与续算数组哈希仍逐项精确校验，只有已经由 S-10 升级到 schema 2 的生成环境元数据允许向前演进。

原失败日志和 JUnit 保留，不覆盖。修正没有修改 `pssolver/`、阈值、skip/xfail、provider identity、checkpoint schema、生产默认值或 S-09 分支。

本记录只授权本地修正和重新执行 CPU 门禁；不授权 push、HPCC 写入、Slurm/H100、tag 或 Release。S-10 的外部复核请求仍为 R-0032。
