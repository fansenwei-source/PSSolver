# PSSolver v0.2.2 本地包准备

日期：2026-10-20

分类：`PASS_V0_2_2_LOCAL_PACKAGE_PREPARATION`

从干净的发布源提交 `2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70`
生成了带 `PSSolver-0.2.2/` 前缀的 Git source archive，并从该归档构建
sdist 与 wheel。再从 sdist 独立重建 wheel 后，263 个 wheel 成员的内容全部
一致；其中 258 个 Python 文件也与干净归档逐字节一致。

本地发行物：

- source archive SHA-256：
  `b441ba1a1a943899da81fb294a67c8e2e69e00117aebc2a26af9d72efa5e41db`
- sdist SHA-256：
  `ad8237e67478f7328c5152981ea4d888f1df64e1af7227f07ba0a0f070b02b39`
- wheel SHA-256：
  `a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e`

独立路径的 installed-wheel 环境不继承 system site packages。它复用已经通过
v0.2.1 资格验证的不可变依赖基线，但只在新环境中移除 0.2.1 并以
`--no-deps` 安装本次 wheel；旧资格环境随后仍能导入 0.2.1。`pip check`、
public version、Functional API 1.0、无源码 shadow import 与 console help 均通过。
N-1 的 13 个 checkpoint 测试和 V-17 的 3 个测试合计 16/16 通过。

V-17 是 host filesystem checkpoint 发布路径修复，不改变求解器、数值算法、
checkpoint bytes/schema、Functional API 或 provider identity，因此不需要 H100
重跑。正式发布前仍须经单独授权，在 HPCC 登录节点对真实 NFS 父目录符号链接
做小型复测。

本提交只登记本地包准备证据，不是发布源，不授权 push、tag、GitHub Release、
PyPI、HPCC 写入或 Slurm/H100 作业。
