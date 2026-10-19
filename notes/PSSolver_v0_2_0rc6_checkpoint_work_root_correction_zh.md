# PSSolver RC6 checkpoint work-root qualification 修正

## 结论

Job `10901114` 已经真实证明 fixed-work pressure smoke 修正通过 H100：环境、7 个
CUDA-only tests、36 个 contract smokes 和 TF32 scope 均通过。累计资格随后在
`current_checkpoints` 阶段失败。

失败不是 checkpoint schema、checksum guard、restore、CUDA 或科学计算错误。Slurm
脚本先在 node-local `/tmp` 成功验证了 `RENAME_NOREPLACE`，也把 `TMPDIR` 固定到了
`/tmp/pssolver_rc6_<job>/pytest_tmp`；但是 stage plan 又把八个 checkpoint cells 的
工作目录显式放回 scratch。scratch 文件系统对该原子发布操作返回 `EINVAL`。

## 修正

本轮只修改 qualification support：

- stage plan 不再传入 scratch `--output-root`，而是冻结
  `--checkpoint-work-root-policy node_local_tmpdir`；
- helper 要求数字形式的 `SLURM_JOB_ID`，并严格核对 `TMPDIR` 等于该 Job 的专属
  `/tmp/pssolver_rc6_<job>/pytest_tmp`；
- 八个 checkpoint cells 运行在 `TMPDIR` 下的 `TemporaryDirectory` 中；
- 临时目录在阶段结束或异常退出时清理；
- scratch 仅保存最终 `current_checkpoints.json` 及常规 runner 日志/进度证据。

未修改 PSSolver runtime、checkpoint schema、完整性门禁、数值算法、阈值、公开 API、
生产默认值、PSSolver-Control 或 nematics3d。本记录不授权 push，也不授权新的 H100
submission。
