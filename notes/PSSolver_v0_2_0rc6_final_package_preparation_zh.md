# PSSolver v0.2.0rc6 最终发布包准备

日期：2026-10-19

分类：`PASS_RC6_FINAL_SDIST_WHEEL_AND_INSTALLED_CPU_RELEASE_PREPARATION`

## 来源

`fix/v0.2.0rc6` 已推送到远端，构建时远端 tip 与本地审查来源均为：

`fd7323d5e9ef87c537ed07476f8bd50b982af443`

构建没有使用脏工作区。源码通过 `git archive` 从该提交导出，归档中嵌入的 commit ID 也精确为 `fd7323d`。

## 最终 artifacts

- source archive：`PSSolver-0.2.0rc6-fd7323d.tar`
  - SHA-256：`c2f3be60f09c47b0aa45064e9ec36a7ad76fde666f207d8595f622057d5b804c`
- sdist：`pssolver-0.2.0rc6.tar.gz`
  - SHA-256：`e6f2013d7d16c3f265759017bae8b5c8d0c1f6a43b54ff52476c9da5c296b907`
- wheel：`pssolver-0.2.0rc6-py3-none-any.whl`
  - SHA-256：`9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5`

三个归档均通过完整性检查。wheel metadata 版本为 `0.2.0rc6`，顶层只包含 `pssolver`、兼容模块和 dist-info。

## Fresh installed-wheel CPU smoke

wheel 使用 `--no-deps` 安装到独立 venv，PSSolver 从该 venv 的 `site-packages` 导入，不存在 source shadow import。

最终结果：

- `pip check`：PASS；
- package version：`0.2.0rc6`；
- functional API：`1.0`；
- console entry point `--help`：PASS；
- Plane public one-step：PASS；
- Periodic continuous/restart：PASS；
- Channel continuous/restart：PASS；
- pytest：3 passed，0 failed，0 skipped，0 xfailed。

第一次 `pip check` 发现宿主继承的 NumPy 2.4.3 与 `nematics3d 0.9.0b1` 的 NumPy 2.3.2 metadata 要求不一致。按照 RC6 已冻结的隔离处理，只在本次 fresh venv 内安装 NumPy 2.3.2，然后从头重跑 `pip check` 和 installed-wheel smoke。没有修改 nematics3d 或宿主环境中的 NumPy。

## 当前边界

发布包准备完成，但尚未：

- 运行最终 installed-wheel H100 release smoke；
- 创建 annotated tag；
- 创建 GitHub prerelease；
- 发布到 PyPI；
- 修改生产默认值。

下一门禁是获得一次明确授权的 RC6 installed-wheel single-H100 release-smoke submission，且不得自动重试。
