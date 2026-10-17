# PSSolver v0.2.0rc5 发布与发布后验证

最终分类：

`PASS_V0_2_0RC5_GITHUB_PRERELEASE_POST_RELEASE_VERIFICATION`

## 发布身份

- 资格化源码：`e2e9d05bfac858d8e1ae1653ac86a2187c9917ef`
- 证据提交：`724686cf890585e059081a289a8dfc6e55f54334`
- annotated tag：`v0.2.0rc5`
- tag object：`d11d4f3c26e851baeb034a3995aa821bfe280fb1`
- tag peeled target：`e2e9d05bfac858d8e1ae1653ac86a2187c9917ef`
- GitHub prerelease：<https://github.com/fansenwei-source/PSSolver/releases/tag/v0.2.0rc5>

tag 精确指向资格化源码，而不是后续 evidence-only commit。GitHub API 的
`target_commitish` 元数据为默认分支 `main`，但发布身份由 annotated tag object 和
peeled target 独立验证。

## 发布资产

- sdist：`pssolver-0.2.0rc5.tar.gz`
  - SHA-256：`4477b6d8b0fa03b4e97390729623322e2b1d6ff453845004598bce4ed30f63dc`
- wheel：`pssolver-0.2.0rc5-py3-none-any.whl`
  - SHA-256：`29333022f799483e177a49d549a8db41d5741ee3c61c24023929f409c8941c39`

GitHub 记录的 digest、上传前正式构件和发布后重新下载的构件三者一致。

## 发布后验证

发布后从 GitHub Release 独立下载两个资产，在 HPCC 新建的 Python 3.10.18 venv 中：

- sdist、wheel 和 installed distribution 的版本均为 `0.2.0rc5`；
- `pip check` 通过；
- PSSolver 从新 venv 的 `site-packages` 导入，没有 source shadow import；
- installed wheel 完成一条 CPU periodic one-step；
- Q、velocity 和 pressure 全部有限；
- 三个最终状态哈希与发布前 package-preparation oracle 逐字节一致。

外部验证器曾出现两次设置错误：sdist metadata 选择器错误，以及未显式设置
`VIRTUAL_ENV` 的 reporter bootstrap 错误。两次均已保留在归档中，没有修改或重建发布
资产，也不支持 solver、数值或科学失败结论。

发布后归档：

`/home/fansenwei/pssolver_v0_2_0rc5_post_release_verification_20261016_v1`

归档校验：`38/38 PASS`；`checksums.sha256` 的 SHA-256 为：

`10ca82b9d7c66d0f7cb35680166c9183067ceedb8b543d8c15525858deef78a0`

## 范围边界

- 本轮没有提交 H100 Job；
- 没有发布到 PyPI；
- 没有修改生产默认值、PSSolver-Control 或 nematics3d；
- batch>1、Plane control、optimizer campaign 和科学控制结论仍未获得授权。
