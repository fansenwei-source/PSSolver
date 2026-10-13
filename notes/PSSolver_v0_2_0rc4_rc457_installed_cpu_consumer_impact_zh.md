# PSSolver v0.2.0rc4 RC4.5.7：installed-wheel Consumer 影响裁决

## 结论

RC4.5.7 已执行，但不能关闭：

`STOPPED_RC4_5_7_CONSUMER_IDENTITY_SCHEMA_MIGRATION_REQUIRED`

Provider `a195bd0` 的 installed wheel TF32 作用域门禁为 6/6 PASS；当前 RC4.4.7 已资格 PSSolver-Control source `0cf55b4` 的完整 installed-wheel CPU suite 为 299 passed、1 failed、1 skipped。唯一 skip 是既有 opt-in R5 benchmark，唯一 failure 是 Periodic adapter 测试仍把 `execution.functional_runtime` 与旧的精确 key 集合比较，没有登记 RC4.5.6 新增的 `tf32_policy`。

## 影响判断

- Consumer adapter 已成功构造 runtime，并完整透传 Provider identity；没有 fallback 或 runtime rejection。
- Channel adapter、checkpoint、adjoint、finite-difference、Taylor 和其他 Consumer 门禁均通过。
- 失败属于 Consumer-owned 精确 schema 预期没有同步，不是 solver、数值、科学或 package failure。
- 不需要改变 `FUNCTIONAL_API_VERSION=1.0`，也不需要提升 Periodic adapter version；新增字段位于 Provider-owned identity 内。
- RC4.5.7 没有修改 PSSolver-Control、`nematics3d` 或生产默认值。

## 所需恢复

PSSolver-Control 应从已登记 RC4.4 pressure 分支创建独立迁移提交：在 Periodic identity 测试中明确登记 `tf32_policy` 的五个字段，并更新兼容性/迁移记录。随后使用新 Consumer commit 与 Provider `a195bd0` 重跑同一 installed-wheel 完整 CPU suite；只有达到 0 failure 才能把 RC4.5.7 标记为完成并进入 RC4.5.8 H100 关闭。

当前没有 H100、push、merge、默认值提升或发布授权。
