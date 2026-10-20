# PSSolver v0.2.0rc6 独立第二方发布后验收登记

## 结论

Claude 于 2026-10-19 完成 read-only、CPU-first 的独立第二方发布后验收，最终分类为：

`PASS_RC6_INDEPENDENT_SECOND_PARTY_POST_RELEASE_ACCEPTANCE`

验收覆盖 Git/Release 身份、发布资产内容、fresh wheel/sdist 安装、公开 API CPU smoke、clean detached tag worktree 完整 CPU suite、既有 H100 证据只读复核及 Codex 结果交叉核对。E1--E5 全部满足，未发现相对 rc5 的 runtime、数值、checkpoint、API 或 packaging 回归。

## 冻结身份

- package source：`fd7323d5e9ef87c537ed07476f8bd50b982af443`
- annotated tag object：`533e3fb4254319d1a6a9a9a0e463447339c18b36`
- GitHub Release：`417781605`
- wheel SHA-256：`9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5`
- sdist SHA-256：`e6f2013d7d16c3f265759017bae8b5c8d0c1f6a43b54ff52476c9da5c296b907`

本登记是 evidence-only successor，不是重新资格化后的 package source，也不要求重新运行 H100。

## 外部证据

验收归档：

`/home/fansenwei/pssolver_v0_2_0rc6_claude_acceptance_20261019_v1`

- checksum：34198/34198 PASS
- `checksums.sha256` SHA-256：`8e3f190aca37113d54414e833ed7e3695b7703040e2a4662e9edde133353ceaa`
- `COMPLETE`：存在
- 独立结果在读取 Codex 记录前已封存于 `a7/independent_results_seal.json`
- clean tag worktree CPU suite：4068 passed，7 deselected，8 subtests passed，0 failed/errors/skipped/xfailed
- H100 release smoke：Job 10901816，COMPLETED，ExitCode 0:0；control 716/716、scratch 14/14 checksum PASS

## 发布决定

保持 v0.2.0rc6 prerelease 不变，不撤回、不替换 tag、Release 或 assets。不开 rc7。最终 v0.2.0 使用 rc6 源码，只允许版本号提升和 CHANGELOG 发布日期更新。

N-1 转入 v0.2.1，在 `maint/v0.2` 上与 Control 研究并行处理。N-1 当前是目标 HPCC NFS 上 checkpoint 发布不可用的问题，但行为为 fail-closed：不会产生错误 checkpoint、不会覆盖已有 checkpoint，也没有数据丢失；当前 Control `research/3d-warmup` 不使用磁盘 checkpoint，也不在 HPCC 运行，因此不阻塞 v0.2.0。

v0.2.0 的已知限制应明确说明临时工作流：在节点本地 `/tmp` 生成 checkpoint，成功后由作业脚本复制到共享存储。该说明不进入 PSSolver 公共 API。

C1-05、C1-10、C1-11、C2-17、C2-18 登记为 0.2.x/0.3 后续工作。

## 范围

本登记没有修改 runtime、数值实现、checkpoint schema、公开 API、默认值、PSSolver-Control、nematics3d、tag、Release 或历史归档；没有 push、PyPI 发布或 H100 提交。
