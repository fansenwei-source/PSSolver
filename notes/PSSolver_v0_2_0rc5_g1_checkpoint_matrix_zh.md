# PSSolver v0.2.0rc5 G-1 发布 checkpoint 矩阵与 Provider identity 影响审查

最终分类：

`PASS_RC5_G1_RELEASED_CHECKPOINT_MATRIX_WITH_PROVIDER_IDENTITY_MIGRATION_REQUIRED`

## Checkpoint 矩阵

- rc1–rc3 的 10 个仓库 fixture 树与只读发布后原始证据逐字节一致；
- 冻结 rc5 installed wheel 的 rc1–rc3 定向矩阵：20 passed，0 failed/skipped；
- 已发布 rc4 wheel 现场写出的 Periodic functional、Channel functional 和 legacy
  Channel checkpoint 均被 rc5 精确接受，且源 checkpoint 未修改；
- rc5 三类 self round-trip 全部精确通过；
- 13 个主要发布矩阵 case 全部符合冻结合同；
- rc1 Periodic、rc1/rc2 Channel functional 的 state-only migration 语义不变；
- rc1 偶数网格 legacy Channel 仍在目标修改前 fail closed。

## Provider identity 影响

Channel identity 未变化：

`3bceb9b2cc1793f1ffa777abcad9d91af4eb6e17da41b7507f26d882259ebd12`

Periodic identity 从：

`abd09b19055cfbac809fa67445a5166d4ee3338e9c439d54da08bd99e105428f`

变为：

`8185c6d4a82d796c9f91857f64d38a9ed7cf6d5641e26423b774c1e561052058`

唯一差异是 `execution.functional_runtime.hermitian_state_projection`，由旧的模糊描述
`self_conjugate_planes_each_step` 更正为
`self_conjugate_stored_planes_projection_each_step.v2`。

已资格化的 rc4 PSSolver-Control wheel 在不施加旧 digest 时能够构造两条 rc5 runtime，
且初态有限；施加冻结的 rc4 expected identity 后，Channel 继续通过，而 Periodic 正确地以
`PSSolverIntegrationUnavailable` fail closed。因此必须先完成 PSSolver-Control Periodic
identity migration 和 installed-wheel requalification，不能直接进入累计 H100 门禁。

## 状态

Checkpoint matrix 与 Provider identity review 已完成，但 `g1_complete = false`，
`eligible_for_h100_submission = false`。下一步是独立 Consumer migration；本轮没有 push、
H100 submission、tag、Release、PyPI、生产默认值或 `nematics3d` 修改。
