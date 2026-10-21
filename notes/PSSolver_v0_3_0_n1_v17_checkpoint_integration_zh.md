# PSSolver 0.3 开发线 N-1 / V-17 checkpoint 集成记录

本地 `dev/0.3` 已移植 v0.2.1 的 NFS 兼容发布路径（N-1）以及 v0.2.2 的父目录符号链接修复（V-17）。版本保持 `0.3.0.dev0`。

集成采用两次定向 cherry-pick 加一次 v0.2.2 evidence 分支合并：开发线的 `checkpoint.py` 与 N-1 原提交的父版本逐字节相同，因此运行时补丁无冲突；随后合并已验收的 v0.2.2 evidence tip，使原始 N-1/V-17 记录成为开发线祖先。冲突仅限版本、开发文档、问题登记表和历史证据测试；保留 0.3 版本与既有开发文档，采用最新版问题登记表，并用前向 successor 记录连接两条证据链。

运行时差异仅为 `pssolver/io/checkpoint.py`。对应行为测试为 `tests/test_v0_2_1_n1_checkpoint_nfs.py` 与 `tests/test_v0_2_2_v17_checkpoint_symlink_parent.py`。checkpoint schema、provider identity、求解方程、数值算法和生产默认值均不改变。

后续 0.3 计划：v0.2.x 登记表中剩余 low/info 项不单独发布 v0.2.3，统一进入 0.3；NEW-14 的 backward TF32 scope 需要 GPU 验证，届时另行申请 GPU/H100 授权。

本记录只描述本地候选；未授权推送，未运行 GPU/H100，也不宣称独立验收已经完成。
