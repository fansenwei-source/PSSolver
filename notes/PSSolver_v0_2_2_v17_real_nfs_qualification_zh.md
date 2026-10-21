# PSSolver v0.2.2 V-17 installed-wheel 与真实 NFS 资格验证

日期：2026-10-20

分类：`PASS_V0_2_2_V17_INSTALLED_WHEEL_AND_REAL_NFS_QUALIFICATION`

本轮在 HPCC 登录节点只执行了一次正式复测，没有提交 Slurm 或 H100 作业，
也没有自动重试。测试使用本地冻结的 v0.2.2 wheel：

`a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e`

测试建立了从 `$HOME` NFS 到 `/scratch1` NFS 的直接父目录符号链接。链接 inode
与解析后目录的设备号分别为 62 和 41；两个文件系统类型都为 `0x6969`。helper
没有强制 fallback，Periodic 与 Channel 的真实 checkpoint export 均各自进入一次
NFS 兼容路径。

两个 family 均满足：

- checkpoint 写入路径保留调用者传入的符号链接拼写；
- checkpoint 导入状态逐位相同；
- 从原状态与恢复状态各续算一步，结果逐位相同；
- completed steps 为 1；
- 没有 staging 临时残留。

远端 home 与 scratch 临时目录在结果取回并完成本地 JSON 校验后全部删除。控制目录
中的 11 项 checksum 全部通过，manifest SHA-256 为：

`62bdd3cbb0acb9ba2bfb62b0ca920d730fa0b7f420240b3fa4a1bdc5bc4f2731`

V-17 的本地 source、installed-wheel、独立第二方复核和 HPCC 真实 NFS 门禁至此
全部通过。该修复只改变 host filesystem checkpoint 发布的设备身份判断，不需要
H100 重跑。

本记录不授权 push、tag、GitHub Release、PyPI、额外 HPCC 写入或 Slurm/H100。
