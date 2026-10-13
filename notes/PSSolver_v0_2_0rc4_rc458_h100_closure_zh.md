# PSSolver v0.2.0rc4 RC4.5.8 累计 H100 关闭

最终分类为
`PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE_WITH_COUNTER_AND_RESERVED_MEMORY_REDUCER_RECOVERY`。

H100 Job `10879606` 完成了 7 个 CUDA-only tests、TF32 scope/caller-state
门禁、96 份 A/B profiles 和 PSSolver-Control installed-wheel H100 suite。Slurm
最终状态为 `FAILED`，原因是科学证据全部产生后，外部 analyzer 把 Channel 的
scheduled refresh calls 错算进 steady-state transform 层；这不是 solver、CUDA、
数值或科学失败。

随后两层 CPU-only recovery 没有重新运行 H100。第一层从 raw `663/2160` 中独立
核对 3 次 scheduled refresh，恢复 base steady-state `11/36`。第二层把独立三 trial
的 CUDA caching-allocator reserved 指标恢复为仓库既有的组级 reducer：
`max(B reserved) / max(A reserved)`。冻结 `1.03` 门限、逐 trial live allocated、
性能和数值身份门禁均未改变。

最终结果为 96/96 profiles、16/16 runtime groups 全部通过；reserved 组级最大比
为 `1.0`。v2、v3、v4 三层归档分别通过 3371/3371、16/16 和 10/10 checksum，
v4 已写入 `COMPLETE`。

因此 B11 转为 `closed_verified`，RC4.5 和 RC4 正确性关闭完成，可以进入独立的
RC4 release preparation。该结论不等于已 merge、已打 tag 或已发布，也不改变
生产默认值、functional API 1.0、checkpoint schema、PSSolver-Control 或
`nematics3d`。

六个受历史内容寻址约束的测试文件通过独立
`historical_successor_sources` 顺序登记到 RC4.5.8 当前版本；RC4.5.1、
RC4.5.5、RC4.5.6 和 RC4.5.7 的既有记录均未改写。
