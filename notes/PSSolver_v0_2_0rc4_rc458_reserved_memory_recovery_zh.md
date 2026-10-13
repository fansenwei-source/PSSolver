# RC4.5.8 reserved-memory reducer recovery

本恢复只修正 qualification analyzer 对 CUDA caching allocator `reserved`
计数的三 trial 汇总方式，不修改求解器、profiler、原始 H100 JSON 或冻结的
`1.03` 门限。

原 analyzer 使用 `max(B_i / A_i)`。三个正式 trial 是独立进程，trial 编号不绑定
allocator segment；同一组 reserved 峰值仅在 trial 2/3 互换时，这个 reducer 会把
allocator 顺序差异误判为显存增长。仓库已有 Plane Nyquist 和 Periodic Hermitian
analyzer 使用稳定的组级定义：

`max(B peak_reserved) / max(A peak_reserved)`。

新的 fail-closed analyzer 恢复这一组级定义，同时保留：

- `peak_reserved` 上限仍为 `1.03`；
- `peak_allocated` 仍逐 trial 严格检查；
- timestep mean、median 和 paired 上限不变；
- initial/final SHA、finite、fallback、graph break 和 transform 门禁不变；
- 原始 96 份 H100 profile 只读且不改写。

在前一轮失败的两个 R512 组中，A/B 的 reserved 三 trial 多重集合完全相同，
组级峰值比均为 `1.0`，且 live allocated 比均为 `1.0`。因此本恢复处理的是
allocator trial-order attribution，不是放宽真实内存回退门禁。

本提交只授权对冻结证据做 CPU-only replay；不授权新的 H100 submission、自动
重试、自动合并、默认提升或生产默认值变更。
