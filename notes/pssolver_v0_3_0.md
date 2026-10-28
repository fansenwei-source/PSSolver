# PSSolver v0.3.0 本地 final 源码冻结候选

S-11 的 `v0.3.0rc1` 候选验收（R-0038）与发布后验收（R-0039）均为
PASS，没有发布阻断项或新发现。S-12 因此在本地冻结 final `v0.3.0`。

相对 rc1 运行时发布源 `c031cdae646edc5b202b05238db1eb513b255805`，
唯一运行时变化是 `pssolver/_version.py` 从 `0.3.0rc1` 提升为 `0.3.0`。
方程、数值算法、checkpoint schema、Functional API 1.1、provider identity、
生产默认值、阈值、skip 和 xfail 均不变。rc1 之后的其他提交只记录候选验收、
发布与发布后验收证据。

本地冻结不授权 push、tag、GitHub Release、PyPI、HPCC、Slurm、GPU 或 H100。
完成干净打包和 CPU/installed-wheel 门禁后，必须先由 R-0040 独立验收；任何
远端发布动作仍需用户逐项授权。
