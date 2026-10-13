# PSSolver v0.2.0rc4 RC4.5.3 checkpoint transactionality

## 结论

`PASS_RC4_5_3_CHECKPOINT_TRANSACTIONALITY_CPU_CANDIDATE`

RC4.5.3 已按 RC4.5.2 冻结合约实现 B7、B24 与 B27。Plane production、legacy Channel production、Periodic production、complete-stress Channel production、finite-Q anchoring、Periodic functional 与 Channel functional 七条 checkpoint 表面现在共享 tensor-free I/O 事务层。

## Schema、路径和恢复

current format 在顶层、嵌套 identity、functional bridge 与 tensor record 上拒绝未登记键。JSON reader 拒绝重复键、NaN/Inf、非 object 根节点以及 metadata symlink。payload 只接受 canonical 单一 basename；绝对路径、`.`、`..`、正反路径分隔符、symlink 与非普通文件都在打开 tensor 前拒绝。

所有 identity、checksum、shape、dtype、finite 与 device materialization 都在第一次 target tensor mutation 前完成。失败不会返回部分 restored state，也不修改 source checkpoint。checkpoint format、tensor layout、科学 identity 与生产默认值均未变化。

## 写入事务与权限

所有 writer 在 target parent 中创建唯一 sibling staging；payload、metadata 和 functional-origin marker 全部成功后只执行一次 atomic directory publish。既有 target 不覆盖，payload、metadata、origin marker 或 publish 失败后 target 与 staging 都不存在。

目录使用 `0o777 & ~caller_umask`，普通文件使用 `0o666 & ~caller_umask`，没有后置 chmod 扩权。`umask 002/022/077` 均有真实文件模式回归。

## 公共异常

新增共享 `CheckpointNotFoundError`、`CheckpointExistsError`、`CheckpointIntegrityError`、`CheckpointCompatibilityError` 与 `CheckpointIOError`。functional 既有异常名称和稳定导出清单保持兼容；functional 各类别同时属于相应 generic category，并继续属于 `FunctionalCheckpointError`。父路径是文件、权限、device-full 和其他 OS failure 进入 I/O 类，原异常通过 `__cause__` 保留，不使用 message text 分类。

## 历史证据和范围

P7.4 历史记录未改写；它继续绑定 RC4.5.3 父提交中的旧 checkpoint source，本 successor 记录绑定当前实现。PSSolver-Control、`nematics3d`、数值热路径、科学模型与生产默认值没有修改；没有运行 H100、push、merge 或发布。

CPU 定向门禁为 215 passed、0 failed；完整门禁为 3276 passed、1 deselected、0 failed、8 subtests passed。唯一 deselect 是要求真实 H100 身份的 CUDA 测试；本地 RTX 3060 Ti 结果未被冒充为 H100 资格证据。没有 skip、xfail 或额外 deselect。

RC4.5.3 完成后具备进入 RC4.5.4 boundary implementation 的资格，但本提交不授权 RC4.5.4、Control 修改、H100、自动 merge、默认提升或发布。
