# successor 70：S-16 中间关

实现提交 `38262ecd066003821ef954a46d12fe7c23bfe4f5` 完成 R7 和 R9：

- R7 把 compact finite canonical JSON、canonical SHA-256、单个/点分 Python
  identifier 的机械实现收敛到标准库-only core helper；调用方继续拥有公开异常类型和
  消息，格式专属 pretty JSON、文件/张量摘要不被错误合并。
- R9 把顶层兼容模块、applications 和 experimental 纳入同一个依赖层棘轮，并把
  `models.active_nematics.stokes -> pssolver.transforms` 改成直接依赖 canonical
  linear-solver/operator 模块。剩余 6 条旧边继续由精确清单锁住，没有移动 runtime
  builder、数值算法或 GPU 专用路径。

R8 此时只完成盘点：18 个测试仍实时依赖本机绝对路径。保留策略草案 D-0015
（同步仓库提交 `73b3db9694edb93acb379a45632dae52a4135d38`）保持
`proposed`，推荐内容寻址的最小 evidence capsule；用户确认之前不实施 R8。
