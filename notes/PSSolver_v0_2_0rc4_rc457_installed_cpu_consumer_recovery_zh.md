# PSSolver v0.2.0rc4 RC4.5.7：installed-wheel Consumer 恢复关闭

RC4.5.7 的恢复门禁已经通过。原始 `STOPPED_RC4_5_7_CONSUMER_IDENTITY_SCHEMA_MIGRATION_REQUIRED` 记录保持原样；本记录只在其后登记 Consumer schema 迁移与 installed-wheel 重验结果。

PSSolver-Control 以两个独立提交完成迁移：`de669fa` 在 Periodic 精确 identity 期望中加入 Provider 的五字段 `tf32_policy`，`b8539e0` 登记该测试的历史 successor。拆分是合理的：第二个提交只维护不可变历史证据链，没有改动 Control 算法、adapter 行为、API 版本、checkpoint schema 或生产默认值。

使用 Provider `a195bd0` 与 Consumer `b8539e0` 构建的两个 wheel 在全新、非 system-site venv 中通过：

- Provider TF32 作用域门禁：6 passed；
- Consumer installed-wheel 完整门禁：310 passed、1 skipped、0 failed；
- 唯一 skip 是既有的 opt-in R5 CUDA benchmark；
- `pip check` 通过；
- `pssolver` 与 `pssolver_control` 均从 venv `site-packages` 导入，没有源码遮蔽；
- `nematics3d` 没有安装，也没有被修改。

PSSolver 本地累计测试也通过：58 个定向测试通过；完整套件为 3362 passed、1 个冻结的 CUDA-only deselect、8 个 subtests passed，0 failed、0 skipped。

`provider_identity.execution.functional_runtime.tf32_policy` 的五个字段与 Provider 公共 identity 一致，canonical SHA-256 为 `f83d7b368bda23aacdf1c70e77de974df1c05a3e8b0c7886191dd82c14ade416`。

因此 RC4.5.7 已完成，Consumer migration 不再阻塞。RC4.5.8 H100 关闭现在具备规划资格，但本次没有获得 H100 或 push 授权，没有提交作业，也没有推送任何分支。
