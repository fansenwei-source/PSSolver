# PSSolver v0.3.0rc1 本地打包冻结

日期：2026-10-28  
分类：`PASS_V0_3_0RC1_LOCAL_PACKAGE_PREPARATION`

发布源码固定为 `c031cdae646edc5b202b05238db1eb513b255805`，其父提交是已经发布到 `dev/0.3` 的 `0e96ee5d05193b0ddae189f6171ba387a8a96b6b`。冻结相对开发线唯一运行时变化是版本号从 `0.3.0.dev0` 变为 `0.3.0rc1`；没有改变方程、数值算法、checkpoint schema、Functional API 1.1、provider identity、生产默认值或资格阈值。

两套 D-0008 基线硬链接克隆都在 CUDA 隐藏并精确 deselect 7 个 CUDA-only node ID 后完成完整 CPU 门禁：各为 `4828 passed, 1 skipped, 7 deselected, 8 subtests passed`。唯一 skip 是未安装可选 `nematics3d` 的解析缺陷测试。v0.2.2 基线克隆本身不含 `matplotlib`，因此只在克隆中补装 `matplotlib 3.10.8` 及依赖；保留的基线没有修改。

干净 `git archive` 构建出：

- source archive：`a8f2c049ef1f46349b30f47bbe917f9382bd4cfccbb8ddbf3261d9191f39b95c`
- wheel：`3e52204593c8e09c7f88cbe54488b977800955f285d90fa1f126bd4dc21c6578`
- sdist：`ab9d5f83408e3591063d08e90df1117fe9f95360ff6a6f646148b2f99e8f367a`

由 sdist 重建的 wheel 与原 wheel 的 222 个成员内容完全一致。wheel 中 217 个 Python 文件均与干净源码归档逐字节一致；S-07 已登记排除的 `pssolver.control` 和 experimental qualification 模块没有重新进入发布包。

安装版 wheel 门禁为 `7 passed`：所有发布 Python 模块可导入（`pygame` 仍是唯一允许缺失的可选依赖），legacy 与 compiled-v2 Channel runtime 都能构造，Plane、Periodic、legacy Channel 和 complete-stress Channel 的公开 dry-run 都产生完整 `public_compilation` 块。安装身份为 `0.3.0rc1`、Functional API `1.1`，`pip check` 通过且无源码遮蔽。

累计门禁 G-02 至 G-05、G-07、G-08 已通过；G-06 依据 D-0013 为有界例外通过。G-01 只等待 R-0038 独立确认 V-44 的前向勘误，G-09 等待同一轮独立候选验收。本提交只是证据登记，不是发布源码。没有授权 push、tag、Release、PyPI、HPCC、Slurm 或 H100。
