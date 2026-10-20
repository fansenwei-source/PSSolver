# PSSolver v0.2.1 N-1：NFS checkpoint 发布本地 CPU candidate

分类：`PASS_V0_2_1_N1_NFS_CHECKPOINT_PUBLICATION_LOCAL_CPU_CANDIDATE`

实现提交：`927dc89aa63bcee9b03248f52662389bc947c47a`

本阶段只实现冻结的 N-1 合同，没有改变求解器方程、数值算法、checkpoint
schema、公共 API、functional API 版本、provider identity、生产默认值或
`nematics3d`。

## 实现

- 支持 `RENAME_NOREPLACE` 的文件系统仍走原快速路径。
- 只有 `renameat2` 返回 `EINVAL`/`EOPNOTSUPP` 且 `statfs` 明确报告
  NFS（`f_type=0x6969`）时才进入兼容路径。
- 兼容路径以 mode `000` 的原子 `mkdir` 占用最终名称，并只允许 rename
  覆盖该发布者自己创建且 inode 身份未变的空占位目录。
- staging 与目标父目录必须位于同一设备。
- rename 响应可能丢失时，回读最终目录并对完整目录结构和所有文件字节重新
  计算摘要；只有与发布前 staging 完全一致才报告成功。
- 无法确认的结果 fail-closed，且不删除不明确的最终目标。
- fallback 后端不写入 checkpoint；快速路径和兼容路径对相同输入生成逐字节
  相同的目录树。

## 本地证据

- N-1 故障注入：13 passed。
- checkpoint/restart 广泛集合（含全部历史 successor 门禁）：350 passed。
- historical-successor 定向收口：185 passed。
- 完整 CPU suite：4114 passed、1 skipped、7 deselected、0 failed、
  8 subtests passed；唯一 skip 是测试环境未安装 optional `nematics3d`，
  7 个 deselect 均为冻结的 CUDA-only node ID。
- 8 个并发兼容路径 writer：恰好 1 个成功，7 个碰撞并 fail-closed。
- 本机 `statfs=0xef53`，即使注入 `EINVAL` 也不会误判为 NFS fallback。

## 尚未完成

本地结果不是 v0.2.1 完成资格。以下门禁仍待后续明确执行：

- installed-wheel CPU；
- HPCC `$HOME` 与 `/scratch1` 的真实 NFS 发布；
- 真实 NFS 目录 fsync 行为持久化；
- 跨客户端并发与丢失确认；
- 最终小型 H100 checkpoint/restart（必须另行授权）。

本阶段未 push、未申请 H100、未创建 tag 或 Release。
