# PSSolver v0.3 S-08 历史 CPU 覆盖返工候选

本记录是 R-0019 之后的前向 successor，不改写原 S-08 候选或 R-0019 结果。

- V-31：fixture manifest 持久化生成环境的 numerical build fingerprint。同一指纹继续要求 rc2 续算逐位一致；不同指纹使用 `rtol=atol=8·eps(dtype)`。rc1 差异必须大于有效舍入容差的 1000 倍。
- V-32：旧版本错误标记为 `exact_legacy_upgrade` 的 rc1 奇数周期轴 checkpoint 会得到专门的格式兼容错误；用户必须从原始 rc1 checkpoint 重新执行 state-only migration。
- A2-25、N5、V-31、V-32 均保持 `claimed_fixed`，等待 R-0020 独立复核后才能转为 `verified`。
- 未修改 checkpoint payload/schema、provider identity、数值算法、阈值或生产默认值；未执行 push、HPCC、GPU 或 H100。
