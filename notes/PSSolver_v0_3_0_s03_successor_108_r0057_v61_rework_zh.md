# V-61 返工

finish_fixed_work_postcondition 显式接收 acceptable；production、transpose、implicit 传入当前值；oracle 显式传入 True，原语义不变。三路故障注入（implicit 30/40/44 次）保留旧异常类型、消息和 termination，不调整旧行为。

新增独立 CPU 故障探针，与父提交/v0.3.0 对称运行；冻结的原四路径矩阵 helper 保持原样。successor 104、106 的绑定测试改为严格前向解析，不改历史记录，不增加 skip/xfail。
