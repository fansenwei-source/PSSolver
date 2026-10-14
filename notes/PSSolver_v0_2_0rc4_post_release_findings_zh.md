# PSSolver v0.2.0rc4 发布后证据登记

## 分类

`FAIL`

本记录只登记证据和 rc5 工作队列，不修改 PSSolver 运行时，不改写任何历史 closure，也不移动 `v0.2.0rc4` tag 或 GitHub Release。

## 决定性回归

B10 从 `closed_verified` 攟为 `regressed_after_closure`。已发布 rc4 wheel 对 rc1、rc2、rc3 正式版本写出的 Periodic 与 Channel functional checkpoint 共六种组合，直接 import 和显式 migration 均拒绝。拒绝是 fail-closed，不会读错状态；但它违反了 RC4.2.4 的跨版本读取承诺。

真实 fixture：

`/home/fansenwei/pssolver_rc4_post_release_verification_20261013_v1/agent_evidence/state/legacy_ckpts_full/`

## 当前矩阵

- `closed_verified`：28；
- `partial_residual`：9；
- `regressed_after_closure`：1；
- `accepted_limitation`：3；
- `deferred_architecture`：9。

九项 residual 为 B7、B11、B13、B17、B20、B23、B24、N3、N6。NEW-1…NEW-15 作为独立 post-release findings 登记，不改变原 50 项历史计数。

B8 仅通过公开 fail-closed restriction 关闭；壁面 lifting 物理没有修复，不得重新开放。B30 的 accepted limitation 保留，但承诺的说明仍待 W1 补充。A3 正确归入 rc3。

## 后续顺序

1. W1 文档；
2. rc5 必须项；
3. rc5 应该项；
4. 可选与发布流程卫生；
5. rc5 冻结后执行累计资格。

H100 只允许在 rc5 冻结后经用户另行授权提交一次。本记录没有提交 H100，没有修改 PSSolver-Control 或 `nematics3d`。

## 验证

- 定向 CPU：101 passed，0 failed，0 skipped，0 xfailed。
- 完整 CPU（CUDA 隐藏）：3415 passed，7 skipped，8 subtests passed，0 failed；skip 数量与 rc4 基线相同，新增 15 项测试全部通过。
- `git diff --check`、严格 JSON 解析与 successor SHA-256 校验均通过。
- 最终 `__pycache__`、`.pyc` 与 `.pytest_cache` 残留为 0。
