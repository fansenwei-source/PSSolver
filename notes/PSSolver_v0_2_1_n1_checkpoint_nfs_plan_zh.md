# PSSolver v0.2.1 N-1：NFS checkpoint 发布计划

状态：`PLAN_V0_2_1_N1_NFS_CHECKPOINT_PUBLICATION_FROZEN`

本计划从本地 v0.2.0 发布准备提交 `6d2934e` 建立在
`maint/v0.2` 上。当前只冻结合同，不修改 checkpoint 实现，不 push，
也不提交 H100 作业。

## 范围

v0.2.1 只修复 N-1：在目标 HPCC 的 NFS `$HOME` 和 `/scratch1` 上，
保留“不覆盖已有 checkpoint”的语义并恢复 checkpoint 发布可用性。
不修改求解器方程、数值算法、checkpoint schema、公共 API、functional API
版本、provider identity、生产默认值或 `nematics3d`。

## 基础合同（18 条）

1. 支持 `RENAME_NOREPLACE` 的文件系统继续使用现有快速路径。
2. 只有明确判定为“不支持该操作”时才进入兼容路径。
3. 权限、I/O、配额、路径、损坏等错误不得触发 fallback。
4. staging 与目标必须位于同一文件系统。
5. 发布前写完全部 payload 和 metadata。
6. 发布前 flush/fsync 全部文件。
7. 发布前 fsync staging 目录。
8. 绝不删除、截断、合并或修改已有目标。
9. 并发写者必须恰有一个成功，其余 fail-closed。
10. 最终目标名下不得出现部分填充的 checkpoint。
11. 持久发布完成前不得报告成功。
12. 文件系统支持时 fsync 目标父目录。
13. 目录 fsync 不受支持必须单独分类并记录。
14. 成功确认后以及可恢复的发布前失败后清理 staging。
15. 传输错误或确认丢失导致结果不明确时，不得删除可能已发布的目标。
16. 两种发布后端生成的 checkpoint 字节、metadata、schema 和 checksum 相同。
17. restart、tamper rejection、旧 checkpoint 和跨设备 identity 行为不变。
18. 测试覆盖成功、碰撞、崩溃边界、权限、I/O 与并发。

## 补充合同

- 兼容路径使用 `mkdir` 原子占位，然后只允许 rename 覆盖该发布者自己创建的
  空占位目录。
- 非幂等重传遇到“可能已经发布但响应丢失”时，必须回读并完整校验；无法确认
  就 fail-closed。
- 在真实 NFS 上测量并持久化目录 fsync 行为。
- publication backend/fallback 标记不得写入 checkpoint。
- 对同一输入，快速路径与兼容路径的 checkpoint tree 必须逐字节一致。
- 必须结合 errno 与 statfs 证据判断“不支持”，并覆盖 raw
  `syscall(316)` 的 `renameat2` 路径。
- 必须做同一 NFS 目标上的跨客户端并发测试。
- provider identity digest 必须保持不变。

H100 只用于最终的小型 checkpoint/restart 作业，而且需要用户另行明确授权。
