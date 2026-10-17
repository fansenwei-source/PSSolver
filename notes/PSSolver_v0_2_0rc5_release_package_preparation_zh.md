# PSSolver v0.2.0rc5 最终发布包准备

最终分类：

`PASS_V0_2_0RC5_SOURCE_AND_INSTALLED_WHEEL_RELEASE_PREPARATION_POST_RELEASE_VERIFICATION_PENDING`

`fix/v0.2.0rc5` 已 push，远端 tip 精确为
`e2e9d05bfac858d8e1ae1653ac86a2187c9917ef`。正式 source archive、sdist 和 wheel
均从该提交的 clean git archive 构建。

正式产物：

- source archive SHA-256：
  `e04d9aef7384dd8d271f59827d3c8b2b9637f50d3050d519c69690496d950299`；
- sdist SHA-256：
  `4477b6d8b0fa03b4e97390729623322e2b1d6ff453845004598bce4ed30f63dc`；
- wheel SHA-256：
  `29333022f799483e177a49d549a8db41d5741ee3c61c24023929f409c8941c39`。

HPCC fresh installed-wheel venv 中 `pip check` 通过，`pssolver` 从 venv 的
`site-packages` 导入，没有 source shadow。公开 Periodic CPU one-step 完成，Q、velocity
和 pressure 全部有限。wheel 中的 `pssolver` payload 与 clean source 逐字节一致；顶层只保留
R5‑4 冻结的兼容模块 `Plane_beris_edwards_stokes.py`。

本机初次环境检查暴露了既有的 `nematics3d 0.9.0b1`–NumPy metadata 冲突，因此本机结果
没有作为发布资格证据，也没有修改 `nematics3d` 或 NumPy。正式门禁改在不安装
`nematics3d` 的 HPCC 固定环境执行并通过。

三个外部 helper 假设问题均已保留：遗漏公开 execution 选项、读取错误的 public result 字段，
以及对既有 wheel/sdist package surface 的过严假设。它们没有导致 wheel 重建，也不是 solver、
数值或科学失败。

最终 scratch 控制目录：

`/scratch1/vincent/PSSolver/pssolver_v0_2_0rc5_release_package_e2e9d05_20261016_v1`

归档为 17/17 checksum PASS，`COMPLETE` 存在。

现在具备 annotated tag 与 GitHub prerelease 的审查资格，但尚未获得创建授权。PyPI 发布仍未
授权；发布后必须继续执行 post-release verification。
