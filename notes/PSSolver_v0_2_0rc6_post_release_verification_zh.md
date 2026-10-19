# PSSolver v0.2.0rc6 独立发布后 CPU 验收登记

最终分类为 `PASS_RC6_INDEPENDENT_POST_RELEASE_CPU_VERIFICATION`。

本轮从 GitHub prerelease 重新下载 sdist 和 wheel，未复用本地构建文件。Release ID 为 `417781605`，annotated tag object 为 `533e3fb4254319d1a6a9a9a0e463447339c18b36`，tag 精确指向正式 package source `fd7323d5e9ef87c537ed07476f8bd50b982af443`。

下载资产的 SHA-256 与 GitHub digest 及发布前冻结值一致。tag、sdist 和 wheel 中 258 个 Python 源文件逐字节一致；两个归档均通过 compileall，wheel 版本和顶层内容正确。

Fresh wheel 与 fresh sdist venv 均以 `--no-deps` 安装 PSSolver，`pip check`、版本 0.2.0rc6、functional API 1.0 和 source-shadow 检查全部通过。NumPy 2.3.2 只安装在两个新 venv 内，没有修改 base Nematics3D 环境或 `nematics3d`。

Installed-wheel CPU smoke 对 Periodic complete-stress 和 Channel complete-stress 各执行一步，均生成 `COMPLETE`，Q、速度和压力全部有限。

第一次直接在 `git archive` 解包目录执行完整 suite，因缺少 `.git` 历史而触发 34 个 failure 和 5 个 error；相关测试都在执行 `git show` 或 `git rev-parse` provenance 合同。该尝试保留并分类为 `EXTERNAL_TEST_HARNESS_MISSING_GIT_PROVENANCE`，不作为产品资格证据。相同 tag source 在 clean detached Git worktree 中重跑后得到 4068 passed、7 deselected、8 subtests passed、0 failed、0 skipped、0 xfailed。七个 deselected CUDA-only tests 已在发布前 Job 10901816 中 7/7 通过。

外部控制目录 `/home/fansenwei/pssolver_v0_2_0rc6_post_release_verification_20261019_v1` 的 checksum 为 10258/10258 PASS，manifest SHA-256 为 `a539e735a520b18b6f6328ce0ac366ccca2831cc80ab8f9ea8492f39d4f2261e`，并已写入 `COMPLETE`。

登记记录的定向合同为 19 passed；登记后的当前分支完整 CPU suite 为 4087 passed、7 deselected、8 subtests passed、0 failed、0 skipped、0 xfailed。

本次登记不修改 tag、Release、runtime、测试、阈值、生产默认值、PSSolver-Control 或 `nematics3d`，不提交 H100，也不发布到 PyPI。RC6 继续保持 prerelease；后续可将本记录交给独立第二方审阅。
