# PSSolver v0.3.0 S-08 历史 CPU checkpoint 覆盖

日期：2026-10-22

分类：`PASS_V0_3_S08_LOCAL_CPU_COVERAGE_CANDIDATE`

本批次补入三份由正式历史版本直接写出的只读 fixture：v0.2.0rc1 与 v0.2.0rc2 的
Channel functional 奇数周期轴 checkpoint，以及 v0.2.0rc1 的真实 Plane lifting
format-1 checkpoint。生成脚本、源码提交、Git archive 哈希和 fixture tree 哈希均固定在
`tests/fixtures/historical_cpu_coverage_v1/manifest.json`。

Channel 的科学边界按真实数据裁决。rc2 奇数周期轴 checkpoint 升级后可逐位精确续算；
rc1 checkpoint 虽可安全恢复状态，但其续算与当前 B2 后的不可压缩动力学不同。因此 rc1
只能声明 `state_only_migration`，不能声明 trajectory-equivalent。该结论防止将缺少的旧版
覆盖误补成不真实的兼容承诺。

真实 rc1 lifted Plane fixture 会在读取 tensor payload 前被明确拒绝；源目录保持不变，目标
目录不存在。S-08 没有重开 lifting，也没有修改 checkpoint payload、schema、provider
identity、数值算法、阈值或生产默认值。

实现提交为 `8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851`。`A2-25` 与 `N5`
当前只标记为 `claimed_fixed`，须由 `R-0019` 独立复核后才能改为 `verified`。
