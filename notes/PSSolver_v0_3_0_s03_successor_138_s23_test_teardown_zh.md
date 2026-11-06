# S-23：动态 Torch 标志的测试收尾兼容

对应 ARCHITECTURE §7。D-0045 的隔离依赖副本保持不变。
首次干净克隆全量为 5648 passed、4 teardown errors（原 7 个 CUDA deselect）。
四个旧 CPU 测试的断言均已通过，错误来自 pytest 9.2 的 MonkeyPatch.undo：
Torch cuBLASModule 用 __getattr__/__setattr__ 暴露 allow_tf32，没有实例字典条目；
MonkeyPatch 在收尾时 delattr，报 AttributeError，而不是恢复原值。

只把这四个测试的两次 flag 赋值改为保存实际 getter 值、注册 setter finalizer，
精确恢复原值。不捕获或忽略错误，不改测试的元数据、调用者状态断言和输入。
两个虚拟属性回归覆盖原值 true/false；AST 守卫确认原断言逐字结构相同。
另对受改动旧文件排序 import，并将两个旧 zip 检查明确为 strict=True。

本后继绑定四个旧测试和新增 helper，以及本批两个测试文件的新版本；
137 和全部历史记录保持原样。原失败日志/JUnit 保留，修正后新提交、新克隆重跑。
不改 pssolver、资格矩阵、基准字节、wheel、依赖、容差、skip/xfail 或 S-22 分支。
123 场景三方重放、VJP、MMS、checkpoint 证据仍绑定 26f88bc 的同一实现。
