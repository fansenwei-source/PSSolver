# PSSolver v0.2.0rc5 G-1 冻结源码与 installed-wheel CPU 关闭记录

最终分类：

`PASS_RC5_G1_FROZEN_SOURCE_AND_INSTALLED_WHEEL_CPU`

本记录登记提交 `674a8121779be72c95aa5dc9e2babca0ce7b93a9` 的冻结源码和
installed-wheel CPU 证据。记录提交只增加证据，不是被资格验证的源码提交，也不要求重跑
已经完成的 CPU 门禁。

## 结果

- 冻结源码完整 suite：3623 passed，8 subtests passed，0 failed/skipped/xfail；
- installed-wheel 定向门禁：178 passed，0 failed/skipped；
- installed-wheel applicable full suite：3609 passed，7 个精确 CUDA-only skip，
  8 subtests passed，0 failed；
- 7 个仅适用于源码布局的本地累计审计测试被明确排除，没有伪装成 skip；
- `pip check`：PASS；
- `pssolver` 从新 venv 的 `site-packages` 导入，source shadow import 为 false；
- source archive SHA-256：
  `76f0a9190255e6e131e820559b3d30bbb51f0ff633c922dda9a49622ed5bc1ee`；
- wheel SHA-256：
  `3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f`；
- 归档 checksum：1712/1712 PASS；manifest SHA-256：
  `6a10414a20d3c5f7c12e21e98afb594fe3008be3010388355a16e6fdb9fbe832`。

第一次 installed-test 调用错误地使用 `/tmp` 作为 cwd，导致 7 个读取仓库相对路径的
静态审计测试出现 `FileNotFoundError`。该次调用被明确排除，不作为资格证据。随后按既有
合同从仓库根目录运行，使用 `python -I`、先导入 installed `pssolver`、再把仓库支持路径
追加到 `site-packages` 之后，并使用 pytest `importlib` 模式；对应 7 个测试先独立通过，
最后完整 applicable suite 从头通过。

## 尚未关闭的 G-1 门禁

1. rc1–rc4 已发布 checkpoint 累计矩阵；
2. Provider identity 影响审查；
3. 如有必要，独立 PSSolver-Control installed-wheel 资格验证；
4. 经单独授权的唯一一次累计 H100 qualification；
5. 发布后验证。

因此 `g1_complete = false`，当前不允许 push、H100 submission、tag、Release 或 PyPI
发布。本轮没有修改生产默认值、PSSolver-Control 或 `nematics3d`。
