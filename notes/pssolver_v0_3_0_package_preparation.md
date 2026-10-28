# PSSolver v0.3.0 本地打包准备

状态：`PASS_V0_3_0_LOCAL_PACKAGE_PREPARATION`，等待 R-0040 独立候选验收。

- 运行时发布源：`47473880534a66f6da940db82122c79cda48a86b`。
- 相对已独立验收的 rc1，运行时只把版本从 `0.3.0rc1` 提升为 `0.3.0`。
- 干净 source archive、sdist 与 wheel 已构建并绑定；从 sdist 重建的 wheel 成员内容逐字节相同。
- wheel 与 rc1 规范化比较后只改变 `_version.py`、`METADATA` 和 `RECORD`。
- 两个 D-0008 基线克隆中的完整 CPU suite 均为 4851 passed、1 个 optional nematics3d skip、7 deselected、8 subtests passed、0 failed。
- installed-wheel 门禁 7/7 通过，pip check 通过，无 source shadow import。
- 没有获得 push、tag、GitHub Release、PyPI、HPCC、Slurm 或 H100 授权。

机器可读的精确哈希、环境和门禁记录见
`notes/pssolver_v0_3_0_package_preparation.json`。
