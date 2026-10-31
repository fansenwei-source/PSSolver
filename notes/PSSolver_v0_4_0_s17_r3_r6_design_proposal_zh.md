# PSSolver v0.4 S-17：R3 / R6 设计提案

状态：`DESIGN_ONLY_AWAITING_USER_DECISION`。本记录只提出方案，没有修改运行时、公开接口、数值实现、checkpoint schema、门限或生产默认值。

## R3：Functional facade 与合同层收敛

推荐方案 A：分阶段退役宽泛兼容 facade。

- v0.4 全程保留 `pssolver.functional.api` 的 Functional API 1.1、导出名称和语义。
- 先把重复的 contracts、errors、validation、versioning 实现收敛到稳定 API 背后，不改变数值路径。
- `pssolver.functional` 宽泛兼容 facade 在 v0.4 中只进入弃用窗口，仍保持可用；不得在没有新的 API 版本决定和 Consumer 资格化之前删除。
- 方案 B 是只收敛内部实现、永久保留两层 facade；风险低，但无法完成 R3 的公共表面收敛。
- 方案 C 是在 v0.4 立即删除兼容 facade 并改变 API 版本；它违反当前 S-17 边界，不推荐。

### 与 PSSolver-Control 的配合

本轮 Control main 的合格来源 `41d1668a7310cc022343ac5917c76234724eeae2` 只从 `pssolver.functional.api` 导入，目标 Provider 是 v0.3.0 发布源 `47473880534a66f6da940db82122c79cda48a86b`。因此方案 A 的 v0.4 内部收敛不应要求 Control 再迁移。

若以后要改变 `pssolver.functional.api` 的名称、语义或版本，必须先在隔离 worktree 中迁移并 CPU 资格化 Control main，再允许 Provider 发布该改变。

`research/3d-warmup` 通过 `pssolver_control.adapters` 使用 Functional API，但当前仍固定在 PSSolver v0.2.0；它不属于本轮 main 迁移，也不得被 R3 顺带迁移。研究分支以后只能通过单独授权、单独证据和单独资格化迁移。

## R6：Robin / finite-Q / lifting 试点表面

推荐方案 A：把 Robin、finite-Q 和 lifting 作为同一个边界试点家族重新评估。

- 三者共同跨越 configuration、planning、operators、runtime 和 workflows；拆开处理会重复迁移所有权和兼容边界。
- 审计期间继续 fail-closed，不开启新的公开执行路径。
- 未资格化入口优先迁入明确的 experimental namespace，并经过弃用窗口后再决定删除；已发布 checkpoint 的只读兼容 reader 必须保留。
- lifting 的 B8/S2-07（壁面物理和运行时 dtype 强转）解决并独立验收前，不得重新开放 lifting。
- 方案 B 只隔离 Robin/finite-Q、继续延期 lifting；范围较小，但会留下重叠迁移。
- 方案 C 立即删除全部试点和 lifting 表面；兼容风险最高，不推荐。

Control 的合格 adapters 不导入这些试点模块，因此 R6 原则上不需要 Control 迁移；如果实施意外触及稳定 Functional API，则必须回到 R3 的 Consumer-first 配合规则。

## 需要用户确认

1. `Q-S17-R3-01`：R3 是否选择 A（推荐：分阶段退役宽泛 facade）？
2. `Q-S17-R3-02`：是否选择 A（推荐：任何未来 Functional API 改变都先迁移 Control main；研究分支另行授权）？
3. `Q-S17-R6-01`：R6 是否选择 A（推荐：Robin、finite-Q、lifting 一起重新评估）？
4. `Q-S17-R6-02`：审计确认入口未资格化时，是否选择 A（推荐：先隔离到 experimental namespace 并弃用，再决定删除）？

用户确认前，不实施 R3 或 R6。
