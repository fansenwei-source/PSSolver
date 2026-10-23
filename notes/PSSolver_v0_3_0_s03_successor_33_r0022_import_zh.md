# R-0022：S-09 H100 前独立复核导入

- `R-0022` 分类为 `REWORK_REQUIRED`；已通过的 execution scope、A·p 缓存、Control
  迁移、installed-wheel、双环境完整 CPU、CPU 遥测和 H100 计划结构证据保持有效。
- 新发现 `V-35`、`V-36` 以 low / registered / 0.3 / S-09 登记。
- `V-35` 要求旧格式迁移和 v2 bridge 在读取张量前固定验证 API `1.0`；`V-36`
  要求兼容 v0.2.x 对 rc1 偶数轴 Channel state-only 迁移使用的历史标签。
- `A3-18` 仍为 `coverage_gap`，`NEW-14` 仍为 `open`，均不提前关闭。
- 下一次独立复核请求为 `R-0023`。本提交仅登记证据，不授权 push、HPCC、GPU 或 H100。
