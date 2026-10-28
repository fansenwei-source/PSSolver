# PSSolver v0.3.0rc1 本地源码冻结候选

日期：2026-10-28

`v0.3.0rc1` 冻结 `dev/0.3` 在 S-03 至 S-10 的累计成果，供 R-0038
独立候选验收。冻结步骤本身只把公开包版本从 `0.3.0.dev0` 提升为
`0.3.0rc1`，并更新发布说明与证据绑定；不改求解器方程、数值算法、
checkpoint schema、Functional API 1.1、provider identity、生产默认值或任何
门限。

51 项统一登记中，46 项已独立验证，2 项按确认决定接受，2 项按确认决定延期。
唯一非终态项 V-44 是封存 S-09 诊断 summary 中手工补全错误提交号的记录问题：
原归档保持不可变，S-09 结案 successor 已把错误值
`616bc453a38b8f1459b1c278b76596f9e83e8687` 向前更正为 Git 可解析的
`616bc453d50040b20d1ef448cf2647358aeee1cc`。rc1 将该现有勘误标为
`claimed_fixed`，必须由 R-0038 验证后才能转为 `verified`。

G-02 至 G-04、G-06 至 G-08 已由既有独立复核和单 H100 累计证据关闭。
G-05 将由本次干净源码、wheel/sdist 和 installed-wheel CPU 门禁完成；G-09
等待 R-0038。此次本地冻结不授权 push、tag、GitHub Release、PyPI、HPCC、
Slurm 或 H100。
