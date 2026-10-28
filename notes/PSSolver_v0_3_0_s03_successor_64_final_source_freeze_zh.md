# S-03 successor 64：v0.3.0 final 源码冻结

R-0038 与 R-0039 已分别通过 `v0.3.0rc1` 候选和发布后独立验收。
本 successor 将相同运行时提升为 final `v0.3.0`：唯一 `pssolver/` 变化是
版本号从 `0.3.0rc1` 变为 `0.3.0`；其余变化为 final 发布说明、冻结记录和
历史测试的前向哈希登记。

本地冻结不改变方程、数值算法、checkpoint schema、Functional API 1.1、
provider identity、生产默认值、阈值、skip 或 xfail，也不授权任何远端操作。
