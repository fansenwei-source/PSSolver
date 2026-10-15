# PSSolver v0.2.0rc5 → PSSolver-Control identity migration 交接合同

## 结论

当前下一步不是继续改 PSSolver，也不是立即提交 H100，而是让独立的 PSSolver-Control 完成一次受控的 Periodic provider-identity 迁移和 installed-wheel CPU 资格验证。

Provider 已冻结的 source 是 `674a8121779be72c95aa5dc9e2babca0ce7b93a9`，版本为 `0.2.0rc5`，qualified wheel SHA-256 为 `3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f`。其 source/full 与 installed-wheel CPU 门禁已经通过；rc1–rc5 checkpoint matrix 也已通过。

Control 的远端基线必须是：

- branch：`migration/pssolver-v0.2.0rc4-rc457-tf32-identity`
- remote tip：`0e3895d664d1b81925127848b2dfde638f39763d`
- 其中真正 qualified Consumer source：`b8539e0ec4c137a6c13746f6832e26e74be24515`

不得从尚未推送的 research 分支开始，不得 squash/rebase 既有资格历史，也不得改写 RC4.5.7 的历史 migration/closure 记录。

## 唯一已知的身份变化

Periodic provider identity 只变化一个路径：

`provider_identity.execution.functional_runtime.hermitian_state_projection`

- rc4：`self_conjugate_planes_each_step`
- rc5：`self_conjugate_stored_planes_projection_each_step.v2`

checkpoint-matrix 诊断构造中观察到的 digest：

- rc4：`abd09b19055cfbac809fa67445a5166d4ee3338e9c439d54da08bd99e105428f`
- rc5：`8185c6d4a82d796c9f91857f64d38a9ed7cf6d5641e26423b774c1e561052058`

这两个 digest **不是可移植的全局 pin**。它们只属于该次诊断使用的具体构造。Claude 必须使用 Control 自己已有的 canonical Periodic fixture，在 Provider `674a812` 上重新计算完整 metadata 和 canonical SHA-256，再更新与该 fixture 精确绑定的期望值。

匹配的 Channel canonical fixture 必须保持完整 metadata 与 digest 不变。诊断构造中的 Channel digest 为 `3bceb9b2cc1793f1ffa777abcad9d91af4eb6e17da41b7507f26d882259ebd12`，同样只可作为交叉检查，不可脱离 fixture 作为全局常量。

## Claude 执行顺序

1. 从上述远端 tip 建立新的 clean worktree/branch，并验证 ancestry 包含 `b8539e0`。
2. 只使用 Provider `674a812` 和指定 wheel SHA；先核对本交接 JSON 中绑定的 Provider records 与外部 manifests。
3. 用 Control 现有 canonical Periodic/Channel fixture 重算 rc4/rc5 identity；保存完整 canonical JSON、SHA-256 和机器可读 structural diff。
4. 要求 Periodic diff 精确只有 Hermitian projection 路径；要求 Channel 完全一致。
5. 只更新面向未来的 identity expectation、compatibility record 和测试；历史 RC4.5.7 记录不改写。除非公开合同测试证明必需，否则不改 adapter/runtime 实现。
6. 加入 fail-closed 测试：旧 Periodic pin 在 mutation/timestep 前拒绝 rc5；新 fixture-specific pin 接受；Channel 继续接受；未提供 expected identity 时 construction 仍可用。
7. 跑 source targeted/full CPU；不得新增 skip/xfail，不得放宽阈值。
8. 从 clean archives 构建 Provider/Consumer wheels，在 fresh venv 中 `--no-deps` 安装，运行 `pip check`，证明无 source shadow import，再跑 installed-wheel targeted/full CPU。
9. 把 source migration 与 CPU qualification 做成可分离的本地 commits，保持 worktree clean，然后停止等待复核。

## 本轮禁止事项

没有授权 push、H100、自动重试、tag、release、PyPI、merge、production default 修改、PSSolver 修改或 nematics3d 修改。

只有 Control CPU migration 和 installed-wheel CPU 证据通过、经复核后，才冻结一次 H100 qualification 合同。当前 `G-1` 仍未完成。

机器可读权威合同见 `notes/PSSolver_v0_2_0rc5_control_identity_migration_handoff.json`。
