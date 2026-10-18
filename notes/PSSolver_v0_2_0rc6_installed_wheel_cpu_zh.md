# PSSolver v0.2.0rc6 frozen installed-wheel CPU 关闭记录

最终分类：

`PASS_RC6_FROZEN_INSTALLED_WHEEL_CPU`

本记录绑定资格源提交 `ea801ec08f5b2465661334e1c126133e55b68e96`。后续记录提交只新增证据，不冒充被验证源码，也不要求重复已经完成的 CPU 门禁。

## 已通过门禁

- 复用 RC6-8 冻结源码完整 suite：3936 passed、7 deselected、8 subtests passed、0 failed/skipped/xfail；
- clean Git archive 成功构建 `0.2.0rc6` sdist 和 wheel；
- fresh venv 使用 `--no-deps` 安装 wheel；
- `pip check`：PASS；
- `pssolver` 从 fresh venv 的 `site-packages` 导入，source shadow import 为 false；
- installed-wheel 定向门禁：166 passed、0 failed/error/skipped；
- installed-wheel applicable full suite：3929 passed、7 个精确 CUDA-only skip、8 subtests passed、0 failed/error；
- 唯一 source-layout-only 文件 `tests/test_phase8_p861_local_cumulative_audit.py` 的 7 个测试被显式排除，没有伪装成 skip 或 deselect；
- 外部归档 checksum：1898/1898 PASS；manifest SHA-256：`e05b1d9d26a898a58e7634505045642c41dee1a6f1ac18a5062d33dba48db6c4`。

## 两个被拒绝的 harness 尝试

第一次错误地把不含 `.git` 的 archive 解包目录作为测试支持根，导致 34 个历史 Git 审计失败和 5 个 error。该结果不作为资格证据。修正后使用 clean、精确 HEAD 的真实 worktree 作为只读测试支持，但在追加该路径前先认证并导入 installed wheel。

第二次虽然测试行为全部通过，但 fresh venv 继承了宿主 NumPy 2.4.3，与既有 `nematics3d 0.9.0b1` 的 `numpy==2.3.2` metadata 冲突，因此 `pip check` 失败，该结果同样不作为资格证据。随后仅在 fresh venv 内安装 NumPy 2.3.2；没有修改 `nematics3d` 或宿主环境。`pip check` 通过后，两套正式测试均从头重跑并通过。

## 尚未关闭

1. released checkpoint compatibility 与 tamper 累计门禁；
2. Provider identity 影响审查；
3. 如影响审查要求，独立 PSSolver-Control migration 与 installed-wheel 资格；
4. 经另行明确授权的 single-H100 累计资格；
5. 独立发布后验收。

因此本阶段没有 push、H100 submission、tag、Release、PyPI 发布或默认值提升，也没有修改 PSSolver-Control 或 `nematics3d`。
